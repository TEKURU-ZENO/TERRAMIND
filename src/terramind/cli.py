"""Command line entry point."""

from __future__ import annotations

import json

import typer

from terramind import registry
from terramind.config import data_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def datasets() -> None:
    """List registered datasets and whether their licence has been verified."""
    for d in registry.load_datasets().values():
        flag = "verified" if d.licence_verified else "UNVERIFIED"
        typer.echo(f"{d.id:<18} {d.modality:<22} {flag:<11} {d.licence}")


@app.command()
def events() -> None:
    """List candidate flood events."""
    for e in registry.load_events().values():
        state = "confirmed" if e.confirmed else "unconfirmed"
        typer.echo(f"{e.id:<14} {e.region:<12} {e.start}..{e.end}  {state}  {e.name}")


@app.command("stac-search")
def stac_search(
    event_id: str,
    collection: str = "sentinel-2-l2a",
    catalog: str = "earth-search",
    max_cloud: float | None = None,
    limit: int = 100,
) -> None:
    """Find scenes covering a flood event and save a provenance manifest."""
    from terramind.ingestion import stac

    event = registry.load_events()[event_id]
    records = stac.search(collection, event.bbox, event.start, event.end, catalog, max_cloud, limit)
    path = stac.write_manifest(
        records,
        data_dir() / "manifests" / f"{event_id}__{collection}__{catalog}.json",
        {
            "event": event_id,
            "collection": collection,
            "catalog": catalog,
            "bbox": event.bbox,
            "start": event.start,
            "end": event.end,
            "max_cloud": max_cloud,
        },
    )
    for r in records:
        cloud = f"{r.cloud_cover:5.1f}%" if r.cloud_cover is not None else "   n/a"
        typer.echo(f"{r.datetime}  cloud {cloud}  {r.id}")
    typer.echo(f"{len(records)} scenes -> {path}")


@app.command("s1f11-download")
def s1f11_download(limit_per_split: int | None = None, s2: bool = False) -> None:
    """Download the Sen1Floods11 hand-labelled chips (446 pairs); --s2 adds optical."""
    from terramind.ingestion import sen1floods11

    registry.require_dataset("sen1floods11")
    typer.echo(json.dumps(sen1floods11.download(limit_per_split, s2=s2)))


@app.command("train-baseline")
def train_baseline(
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    crop: int = 256,
    seed: int = 0,
    modality: str = "s1",
    holdout_event: str | None = None,
    modality_dropout: float = 0.0,
    arch: str = "unet",
) -> None:
    """Train a U-Net flood baseline (modality: s1, s2 or s1s2) and write report.json.

    --holdout-event removes one flood event from training entirely
    (leave-one-event-out) and scores the model on it.
    """
    from terramind.training import train_baseline as tb

    out = tb.run(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        crop=crop,
        seed=seed,
        modality=modality,
        holdout_event=holdout_event,
        modality_dropout=modality_dropout,
        arch=arch,
    )
    typer.echo(f"report: {out / 'report.json'}")


@app.command("eval-ensemble")
def eval_ensemble(modality: str = "s1", epochs: int = 30, common_mask: bool = False) -> None:
    """Evaluate the seed ensemble: IoU, calibration and selective prediction."""
    from terramind.training import eval_ensemble as ee

    out = ee.run(modality=modality, epochs=epochs, common_mask=common_mask)
    report = json.loads(out.read_text(encoding="utf-8"))
    for split, r in report["results"].items():
        for name in ("single_mean", "ensemble", "ensemble_temp_scaled"):
            m = r[name]
            iou = f"IoU {m['iou']:.3f}  " if "iou" in m else " " * 11
            typer.echo(
                f"{split:<8}{name:<22}{iou}ECE {m['ece']:.4f}  NLL {m['nll']:.4f}  "
                f"Brier {m['brier']:.4f}  err@100/90/80 "
                f"{m['error_at_100pct_coverage']:.4f}/{m['error_at_90pct_coverage']:.4f}/"
                f"{m['error_at_80pct_coverage']:.4f}"
            )
    typer.echo(f"temperatures {report['temperature']} -> {out}")


@app.command("build-explorer")
def build_explorer(modality: str = "s1", epochs: int = 30) -> None:
    """Precompute map overlays and the chip index used by the Earth Explorer."""
    from terramind.explorer import build

    typer.echo(f"index: {build.build(modality=modality, epochs=epochs)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API on localhost."""
    import uvicorn

    uvicorn.run("terramind.api.app:app", host=host, port=port)


@app.command("build-evidence")
def build_evidence() -> None:
    """Rebuild the evidence corpus from run reports, the chip index, manifests and docs."""
    from collections import Counter

    from terramind.evidence import store

    records = store.build_corpus()
    path = store.save_corpus(records)
    typer.echo(f"{len(records)} records {dict(Counter(r.kind for r in records))} -> {path}")


@app.command("rag-benchmark")
def rag_benchmark(k: int = 10) -> None:
    """Compare vector-only, spatial-only and hybrid retrieval on generated queries."""
    from terramind.evidence import store
    from terramind.rag import benchmark
    from terramind.rag.retriever import Retriever
    from terramind.training.runs import runs_dir

    result = benchmark.run(Retriever(store.load_corpus()), k=k)
    typer.echo(f"embedder {result['embedder']}; cases {result['cases']}")
    typer.echo(f"{'group':<13}{'mode':<16}{'P@k':>7}{'R@k':>7}{'nDCG':>7}{'in-region':>11}")
    for group, modes in result["results"].items():
        for mode, m in modes.items():
            region = f"{m['spatial_ok']:.2f}" if "spatial_ok" in m else "-"
            typer.echo(
                f"{group:<13}{mode:<16}{m['precision']:>7.2f}{m['recall']:>7.2f}"
                f"{m['ndcg']:>7.2f}{region:>11}"
            )
    out = runs_dir() / "rag-benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    typer.echo(f"-> {out}")


@app.command()
def ask(question: str, llm: bool = False, trace: bool = False) -> None:
    """Ask TerraCore a question. Every statement in the answer cites evidence."""
    from terramind.agents.llm import OllamaLLM
    from terramind.agents.workflow import TerraCore
    from terramind.evidence import store
    from terramind.rag.retriever import Retriever

    core = TerraCore(Retriever(store.load_corpus()), llm=OllamaLLM() if llm else None)
    report = core.ask(question)
    typer.echo(f"[{report['verdict']}]")
    typer.echo(report["answer"])
    for caveat in report["caveats"]:
        typer.echo(f"  caveat: {caveat}")
    for e in report["evidence"]:
        typer.echo(f"  [{e['n']}] {e['id']}  (provenance {e['provenance']})")
    if trace:
        for step in report["trace"]:
            typer.echo(f"  {step['agent']}.{step['action']}: {json.dumps(step['detail'])[:300]}")


@app.command("build-sim")
def build_sim(modality: str = "s1") -> None:
    """Turn benchmark chips and ensemble predictions into simulation scenarios."""
    from terramind.sim import scenarios

    path = scenarios.build(modality)
    data = scenarios.load(modality)
    counts = {s: len(data.select(s)) for s in ("train", "valid", "test", "bolivia")}
    typer.echo(f"{len(data)} chips, usable scenarios {counts} -> {path}")


@app.command("sim-train")
def sim_train(
    modality: str = "s1",
    seed: int = 0,
    updates: int = 400,
    uncertainty: bool = True,
    task: str = "response",
    waste_cost: float = 1.0,
    imitate_batches: int = 0,
    critic_warmup: int = 0,
    holdout: int = 0,
    lr: float = 3e-4,
    entropy: float = 0.01,
    name: str = "",
    critic: str = "local",
) -> None:
    """Train a PPO policy. --task dispatch uses the harder task with a waste cost."""
    from terramind.sim import ppo

    out = ppo.train(
        ppo.PPOConfig(
            modality=modality,
            seed=seed,
            updates=updates,
            use_uncertainty=uncertainty,
            task=task,
            waste_cost=waste_cost,
            imitate_batches=imitate_batches,
            critic_warmup=critic_warmup,
            holdout=holdout,
            lr=lr,
            entropy=entropy,
            name=name,
            critic=critic,
        )
    )
    typer.echo(f"run: {out}")


@app.command("sim-eval")
def sim_eval(modality: str = "s1") -> None:
    """Compare heuristic, optimisation and PPO policies on fixed simulated episodes."""
    from terramind.sim import evaluate

    out = evaluate.run(modality)
    report = json.loads(out.read_text(encoding="utf-8"))
    typer.echo("SIMULATED results. auc = mean fraction of need served over the episode.")
    typer.echo(f"exploration weight tuned on valid: {report['beta_tuning']}")
    for split, r in report["results"].items():
        typer.echo(f"\n{split} ({next(iter(r['policies'].values()))['scenarios']} scenarios)")
        for name, m in r["policies"].items():
            spread = f"  seed std {m['auc_seed_std']:.3f}" if m.get("auc_seed_std") else ""
            typer.echo(
                f"  {name:<24}auc {m['auc']:.3f} [{m['auc_ci95'][0]:.3f}, {m['auc_ci95'][1]:.3f}]"
                f"  served {m['served']:.3f}{spread}"
            )
        for name, d in r["paired_auc"].items():
            typer.echo(f"  {name:<40}{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    typer.echo(f"-> {out}")


@app.command("chain-ablation")
def chain_ablation() -> None:
    """E8: decision quality as each link of the chain is added (simulated)."""
    from terramind.sim import chain

    out = chain.run()
    report = json.loads(out.read_text(encoding="utf-8"))
    typer.echo("SIMULATED. Fixed lookahead planner; only its input changes.")
    for setting, r in report["settings"].items():
        typer.echo(
            f"\n{setting} {r['config']}  uncertainty weight {r['uncertainty_weight']['chosen']}"
        )
        names = list(next(iter(r["splits"].values()))["stages"])
        for name in names:
            cells = []
            for split, s in r["splits"].items():
                m = s["stages"][name]
                lo, hi = m["vs_reference_ci95"]
                cells.append(
                    f"{split} {m['auc']:.3f} ({m['vs_reference']:+.3f} [{lo:+.3f},{hi:+.3f}])"
                )
            typer.echo(f"  {name:<34}" + "   ".join(cells))
    typer.echo(f"-> {out}")


@app.command()
def report(chip_id: str, llm: bool = False) -> None:
    """Situation report for one chip: perception, reliability and a simulated response.
    --llm adds a local LLM's plain-language summary, kept only if it adds no numbers."""
    from terramind.integration import situation

    r = situation.build(chip_id)
    if llm:
        s = situation.narrate(r)
        typer.echo(f"summary: {s['text']}" if s["accepted"] else f"summary dropped: {s['reason']}")
    typer.echo(f"{r['chip']} ({r['event']}, {r['split']})  reliability: {r['reliability']}")
    typer.echo(f"recommendation: {r['recommendation']}")
    for flag in r["flags"]:
        typer.echo(f"  flag: {flag}")
    for section in r["sections"]:
        typer.echo(f"\n{section['title']}")
        for st in section["statements"]:
            mark = " [simulated]" if st["simulated"] else ""
            typer.echo(f"  - {st['text']}{mark}  <{', '.join(st['evidence'])}>")
    for caveat in r["caveats"]:
        typer.echo(f"  caveat: {caveat}")


@app.command("agents-eval")
def agents_eval(llm_model: str = "llama3:8b", no_llm: bool = False, split: str = "dev") -> None:
    """RQ5: compare TerraCore, its first version, a no-critic ablation, a fixed pipeline and
    one LLM agent. --split test runs the held-out questions."""
    from terramind.agents import evaluate

    out = evaluate.run(None if no_llm else llm_model, split)
    report = json.loads(out.read_text(encoding="utf-8"))
    typer.echo(
        f"{report['split']}: {len(report['questions'])} questions, fingerprint {report['fingerprint']}"
    )
    for name, s in report["systems"].items():
        m = s["summary"]
        typer.echo(
            f"{name:<26} accuracy {m['accuracy']:.2f}  answerable {m['answerable_correct']:.2f}  "
            f"refused unanswerable {m['unanswerable_refused']:.2f}  "
            f"unsupported decimals {m['unsupported_decimal_rate']:.2f}  "
            f"{m['seconds_per_question']:.1f}s/q"
        )
        typer.echo("    " + "  ".join(f"{c} {v:.2f}" for c, v in s["by_category"].items()))
    typer.echo(f"-> {out}")


@app.command("dispatch-eval")
def dispatch_eval(modality: str = "s1") -> None:
    """E9: the harder simulated task, where deploying to a dry cell costs something."""
    from terramind.sim import dispatch

    out = dispatch.run(modality)
    report = json.loads(out.read_text(encoding="utf-8"))
    typer.echo("SIMULATED. utility = (flooded area served - cost x wasted deployments) / total.")
    for split, skills in report["cell_skill"].items():
        cells = ", ".join(
            f"{n}: Brier {s['brier']:.4f} AUC {s['auc']:.3f}" for n, s in skills.items()
        )
        typer.echo(f"dry-cell prediction on {split}: {cells}")
    for cost, r in report["settings"].items():
        typer.echo(f"\nwaste cost {cost}")
        for name in next(iter(r["splits"].values()))["policies"]:
            cells = []
            for split, s in r["splits"].items():
                m = s["policies"][name]
                lo, hi = m["vs_reference_ci95"]
                cells.append(
                    f"{split} {m['utility']:.3f} ({m['vs_reference']:+.3f} [{lo:+.3f},{hi:+.3f}]) "
                    f"waste {m['wasted_deployments']:.1f}/{m['deployments']:.0f}"
                )
            typer.echo(f"  {name:<27}" + "   ".join(cells))
    typer.echo(f"-> {out}")


@app.command("fusion-eval")
def fusion_eval() -> None:
    """E1: late fusion, modality dropout and fused models with a sensor missing."""
    from terramind.training import fusion

    out = fusion.run()
    report = json.loads(out.read_text(encoding="utf-8"))
    typer.echo(
        f"late fusion optical weight (tuned on valid): {report['late_fusion_optical_weight']}"
    )
    for name, r in report["results"].items():
        typer.echo(
            f"  {name:<38}test IoU {r['test']['iou']:.3f}  Bolivia IoU {r['bolivia']['iou']:.3f}"
        )
    typer.echo(f"-> {out}")


@app.command("forecast-fetch")
def forecast_fetch() -> None:
    """Download GloFAS discharge and ERA5 precipitation at the event sites (Open-Meteo)."""
    from terramind.ingestion import openmeteo

    typer.echo(openmeteo.fetch())


@app.command("forecast")
def forecast() -> None:
    """E3: forecast discharge and flood-threshold exceedance 1, 3 and 7 days ahead."""
    from terramind.forecast import discharge

    out = discharge.run()
    report = json.loads(out.read_text(encoding="utf-8"))
    for lead, r in report["leads"].items():
        nse = "  ".join(f"{m} {v:.3f}" for m, v in r["median_nse"].items())
        typer.echo(f"lead {lead} d  median NSE: {nse}")
    typer.echo(f"-> {out}")


@app.command("roads")
def roads(fetch: bool = True) -> None:
    """E10: OSM road graphs on the chips and a graph network for flooded road points."""
    from terramind.roads import gnn, graph, osm

    if fetch:
        osm.fetch()
    graph.build_all()
    out = gnn.run()
    report = json.loads(out.read_text(encoding="utf-8"))
    for split, r in report["results"].items():
        t = r["threshold"]["f1"]
        typer.echo(
            f"{split}: F1 threshold {t:.3f}  mlp {r['mlp']['f1_mean']:.3f}  gnn {r['gnn']['f1_mean']:.3f}"
        )
    typer.echo(f"-> {out}")


@app.command("db-load")
def db_load() -> None:
    """Load chips, scenes and evidence into PostGIS (TERRAMIND_DATABASE_URL) or DuckDB."""
    from terramind.storage.spatial import open_store

    store = open_store()
    typer.echo(f"{type(store).__name__}: {store.load()}")


@app.command("graph-load")
def graph_load() -> None:
    """Load the evidence graph into Neo4j (TERRAMIND_NEO4J_URL, TERRAMIND_NEO4J_PASSWORD)."""
    from terramind.storage.graphdb import EvidenceGraph

    g = EvidenceGraph()
    try:
        typer.echo(g.load())
    finally:
        g.close()


@app.command("publish-objects")
def publish_objects() -> None:
    """Upload map layers and run reports to MinIO (TERRAMIND_S3_*) or the local object store."""
    from terramind.storage.objects import publish

    typer.echo(publish())


@app.command()
def validate(figures: bool = True) -> None:
    """Collect all results, check the protocol, and write docs/results.md and figures."""
    from terramind.validation import report as rep

    out = rep.run(figures=figures)
    for c in out["claims"]:
        typer.echo(f"{c['rq']}: {c['verdict']}")
    for c in out["checks"]:
        typer.echo(f"{'pass' if c['ok'] else 'FAIL'}  {c['check']} ({c['detail']})")
    typer.echo(f"-> {out['markdown']}, {out['json']}, {len(out['figures'])} figures")


if __name__ == "__main__":
    app()
