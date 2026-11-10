# TERRA MIND

Uncertainty-aware, evidence-grounded geospatial decision intelligence.
First research track: **global flood intelligence**.

This repository is at **Sprint 0**: a reproducible data foundation and a first
flood-segmentation baseline. Agents, RAG, simulation, RL and the dashboard are
deliberately not here yet. The full specification is in
[docs/specification.md](docs/specification.md).

## What exists

| Piece | Where | Status |
|---|---|---|
| Dataset and licence registry | `configs/datasets.yaml`, `src/terramind/registry.py` | Licences recorded but **not yet verified** |
| Flood events and geographic splits | `configs/events.yaml` | Candidate events; Sen1Floods11 split with Bolivia held out |
| STAC scene discovery | `src/terramind/ingestion/stac.py` | Sentinel-1/2, Copernicus DEM |
| Sen1Floods11 downloader | `src/terramind/ingestion/sen1floods11.py` | Hand-labelled subset |
| SAR flood baseline (U-Net) | `src/terramind/models/unet.py`, `src/terramind/training/` | Trains on an 8 GB GPU |
| Threshold baselines | `experiments/baselines/threshold_baseline.py` | Otsu and tuned global threshold |
| Ensemble uncertainty | `src/terramind/uncertainty/`, `terramind eval-ensemble` | ECE, NLL, Brier, selective error |
| API | `src/terramind/api/app.py` | Read-only; spatial filter in-process |
| Earth Explorer | `apps/web` (Next.js + MapLibre) | Benchmark chips and event scenes |
| Evidence store and hybrid retrieval | `src/terramind/evidence`, `src/terramind/rag` | In-process (Qdrant local, Shapely, NetworkX) |
| TerraCore agents | `src/terramind/agents` (LangGraph) | Tool-grounded, critic-verified, no LLM required |
| PostGIS + MinIO | `docker-compose.yml` | Defined, not wired into code yet |

## Setup (Windows, Python 3.11)

```bash
py -3.11 -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

PyTorch comes from the CUDA 12.8 index (required for RTX 50-series):

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
.venv/Scripts/terramind datasets
```

```bash
.venv/Scripts/terramind stac-search chennai-2015 --collection sentinel-2-l2a --max-cloud 40
```

```bash
.venv/Scripts/terramind s1f11-download
```

```bash
.venv/Scripts/terramind train-baseline --epochs 30
```

Train on optical or fused inputs, aggregate seeds, and evaluate the ensemble:

```bash
.venv/Scripts/terramind s1f11-download --s2
```

```bash
.venv/Scripts/terramind train-baseline --modality s1s2 --seed 0
```

```bash
.venv/Scripts/python experiments/baselines/aggregate.py
```

```bash
.venv/Scripts/terramind eval-ensemble --modality s1
```

## Earth Explorer

A map of every benchmark chip with radar/optical imagery, labels, ensemble
predictions, errors and member disagreement, plus scene footprints for the
candidate events. Build the overlays once, then run the API and the web app:

```bash
.venv/Scripts/terramind build-explorer
```

```bash
.venv/Scripts/terramind serve
```

```bash
npm --prefix apps/web run dev
```

Then open http://localhost:3000. The web app proxies `/api` to the Python API
on port 8000.

## Ask TerraCore

Evidence-grounded question answering. Build the evidence corpus, then ask from
the command line or the Ask tab of the explorer:

```bash
.venv/Scripts/terramind build-evidence
```

```bash
.venv/Scripts/terramind ask "How well does the flood model perform in Bolivia?" --trace
```

```bash
.venv/Scripts/terramind rag-benchmark
```

Answers are assembled from stored evidence by fixed rules and checked by a
critic; the system abstains when nothing supports an answer. `--llm` lets a
local Ollama model reword a verified answer.

## TerraSim (simulated flood response)

A simulated decision problem built on the real chips: three teams move over a
16 x 16 grid of a chip for 48 steps and are rewarded for reaching flooded
cells early. They see the model's prediction, not the labels. The flood extent
and predictions are real; the teams and the reward are invented.

```bash
.venv/Scripts/terramind build-sim --modality s1
```

```bash
.venv/Scripts/terramind sim-train --seed 0
```

```bash
.venv/Scripts/terramind sim-eval --modality s1
```

`sim-eval` compares random, greedy, assignment (Hungarian), lookahead planning,
an oracle that sees the truth, and any trained PPO policies on fixed episodes,
and writes `experiments/sim/runs/e7-<modality>.json`. The Simulate tab of the
explorer replays an episode on a chip.

## End-to-end chain

```bash
.venv/Scripts/terramind chain-ablation
```

```bash
.venv/Scripts/terramind report Bolivia_242570
```

`chain-ablation` (E8) measures simulated decision quality as the planner is
given nothing, cruder or better flood maps, or the truth; it needs
`build-sim` for `s1`, `s2` and `s1s2`. `report` runs the whole chain for one
chip (perception, reliability from evidence, simulated response) with evidence
ids on every statement; the chip card in the explorer shows the same report.
Run `build-evidence` after `sim-eval` or `chain-ablation` so their results
can be cited.

Each training run writes `report.json` (config, git commit, per-split and
per-event IoU/F1) under `experiments/baselines/runs/`.

## Layout

```
configs/       dataset registry, flood events, splits
src/terramind/ modular monolith: ingestion, processing, models, training,
               uncertainty, explorer, api
apps/web/      Earth Explorer front end
experiments/   baselines, ablations, transfer, evaluation
research/      literature notes, methodology, results, paper
docs/          specification and weekly research log
tests/
```

Split into services only when a component needs its own compute.

## Rules this repo follows

- No random patch splits. Test regions are geographically unseen (Bolivia for
  Sen1Floods11).
- Metrics are reported per event, not only pooled over pixels.
- Every dataset has a registry entry with source and licence before it is used.
- Simulated outcomes are never reported as observed outcomes.
