import pytest

from terramind.evidence.store import Evidence
from terramind.rag.query import build_gazetteer, parse
from terramind.rag.retriever import HashingEmbedder, Retriever, spatial_score, temporal_score

pytest.importorskip("langgraph")
from terramind.agents.workflow import Claim, TerraCore

KERALA = [76.2, 9.4, 76.8, 10.3]
BOLIVIA = [-65.5, -15.0, -64.5, -13.0]


def corpus() -> list[Evidence]:
    def scene(i, day, cloud, bbox=KERALA, event="kerala-2018"):
        return Evidence(
            id=f"scene:{i}",
            kind="scene",
            title=f"sentinel-2-l2a scene {day}",
            text=f"sentinel-2-l2a optical satellite scene acquired {day}. Cloud cover {cloud}%.",
            source={"type": "stac"},
            provenance=0.9,
            bbox=bbox,
            start=day,
            end=day,
            links=[f"event:{event}"],
            data={
                "collection": "sentinel-2-l2a",
                "event": event,
                "cloud_cover": cloud,
                "platform": "s2a",
            },
        )

    return [
        Evidence("event:kerala-2018", "event", "Candidate flood event: Kerala floods",
                 "Kerala floods 2018, unconfirmed.", {"type": "config"}, 0.4, KERALA,
                 "2018-08-08", "2018-08-31", data={"event": "kerala-2018"}),
        Evidence("event:s1f11-Bolivia", "event", "Sen1Floods11 flood event: Bolivia",
                 "Bolivia benchmark event.", {"type": "dataset"}, 1.0, BOLIVIA,
                 data={"event": "Bolivia"}),
        scene("a", "2018-08-10", 80.0),
        scene("b", "2018-08-22", 10.0),
        scene("c", "2019-03-01", 5.0),
        scene("far", "2018-08-12", 5.0, bbox=[6.8, 50.4, 7.3, 50.6], event="ahr-2021"),
        Evidence("chip:Bolivia_1", "chip", "chip", "Bolivia chip.", {"type": "raster"}, 1.0,
                 [-65.0, -14.0, -64.95, -13.95], links=["event:s1f11-Bolivia"],
                 data={"event": "Bolivia", "split": "bolivia", "iou": 0.5, "water_fraction": 0.25}),
        Evidence("benchmark:s1:event:Bolivia", "benchmark", "U-Net radar on Bolivia",
                 "On the Bolivia flood event the radar U-Net reaches water IoU 0.628.",
                 {"type": "runs"}, 1.0, BOLIVIA, links=["event:s1f11-Bolivia"],
                 data={"modality": "s1", "event": "Bolivia", "iou_mean": 0.628, "seeds": 5,
                       "test_chips": 15, "seen_in_training": False}),
        Evidence("dataset:openstreetmap", "dataset", "Dataset: OpenStreetMap",
                 "OpenStreetMap roads. Licence: ODbL 1.0, unverified.", {"type": "config"}, 0.5,
                 data={"dataset": "openstreetmap", "licence_verified": False}),
    ]  # fmt: skip


@pytest.fixture(scope="module")
def core() -> TerraCore:
    return TerraCore(Retriever(corpus(), embedder=HashingEmbedder()))


def test_parser_resolves_place_time_and_intent():
    q = parse("Which Sentinel-2 scenes cover Kerala in August 2018?", build_gazetteer(corpus()))
    assert [r.evidence_id for r in q.regions] == ["event:kerala-2018"]
    assert (q.start, q.end) == ("2018-08-01", "2018-08-31")
    assert "scene" in q.kinds and q.modalities == ["s2"]


def test_parser_does_not_invent_unknown_places():
    assert parse("Flood risk in Mumbai?", build_gazetteer(corpus())).regions == []


def test_score_components():
    assert spatial_score(KERALA, [KERALA]) == 1.0
    assert 0 < spatial_score(BOLIVIA, [KERALA]) < 0.01
    assert spatial_score(None, [KERALA]) == 0.0
    assert temporal_score("2018-08-10", "2018-08-10", "2018-08-01", "2018-08-31") == 1.0
    assert 0 < temporal_score("2019-03-01", "2019-03-01", "2018-08-01", "2018-08-31") < 0.2


def test_hybrid_ranks_in_region_in_window_scenes_first(core):
    q = parse("Which Sentinel-2 scenes cover Kerala in August 2018?", core.gazetteer)
    top = [h.evidence.id for h in core.retriever.retrieve(q, k=3, kinds=["scene"])]
    assert set(top[:2]) == {"scene:a", "scene:b"}


def test_scene_answer_counts_only_matching_scenes_and_cites_them(core):
    report = core.ask("Which Sentinel-2 scenes cover Kerala in August 2018?")
    assert report["verdict"] == "answer"
    assert report["answer"].startswith("- 2 sentinel-2-l2a scenes")
    assert {e["id"] for e in report["evidence"]} == {"scene:a", "scene:b"}


def test_time_window_is_relaxed_once_and_flagged(core):
    report = core.ask("Which Sentinel-2 scenes cover Kerala in January 2015?")
    assert report["verdict"] == "answer"
    assert any("time window" in c for c in report["caveats"])


def test_performance_answer_uses_benchmark_number(core):
    report = core.ask("How well does the model perform in Bolivia?")
    assert (
        "0.628" in report["answer"] and report["evidence"][0]["id"] == "benchmark:s1:event:Bolivia"
    )


def test_abstains_on_unknown_place_and_unsupported_task(core):
    report = core.ask("Which roads in Mumbai will flood in the next 48 hours?")
    assert report["verdict"] == "abstain" and report["evidence"] == []
    assert "not built yet" in report["answer"]


def test_low_provenance_is_flagged(core):
    report = core.ask("What is the licence for OpenStreetMap data?")
    assert report["evidence"][0]["id"] == "dataset:openstreetmap"
    assert any("unverified" in c.lower() for c in report["caveats"])


def test_critic_rejects_unsupported_claims(core):
    state = {
        "query": parse("x", core.gazetteer),
        "attempts": 2,
        "claims": [
            Claim("t", "IoU is 0.99", ["benchmark:s1:event:Bolivia"], numbers=[0.99]),
            Claim("t", "3 scenes", ["scene:a"], count=3),
            Claim("t", "made up", ["scene:does-not-exist"]),
            Claim(
                "t", "mean water 90%", ["chip:Bolivia_1"], means=[("water_fraction", 100.0, 90.0)]
            ),
            Claim("t", "IoU is 0.628", ["benchmark:s1:event:Bolivia"], numbers=[0.628]),
        ],
    }
    out = core.critic(state)
    assert [c.text for c in out["claims"]] == ["IoU is 0.628"]
    assert len(out["rejected"]) == 4 and out["verdict"] == "answer"


def test_llm_rewording_is_dropped_if_it_adds_numbers():
    class Liar:
        def rewrite(self, question, findings):
            return "The model reaches IoU 0.95 in Bolivia."

    class Faithful:
        def rewrite(self, question, findings):
            return "In Bolivia the radar model reaches IoU 0.628 [1]."

    retriever = Retriever(corpus(), embedder=HashingEmbedder())
    q = "How well does the model perform in Bolivia?"
    assert TerraCore(retriever, llm=Liar()).ask(q)["llm_polished"] is False
    assert TerraCore(retriever, llm=Faithful()).ask(q)["llm_polished"] is True


def sim_corpus() -> list[Evidence]:
    def policy(name, auc):
        return Evidence(
            f"simulation:e7:test:{name}", "simulation", f"Simulated: {name}",
            f"Simulated: on the test split the {name} policy reaches on average {auc:.3f}.",
            {"type": "simulation"}, 0.7,
            data={"simulated": True, "split": "test", "policy": name, "auc": auc},
        )  # fmt: skip

    return [
        *corpus(),
        policy("lookahead", 0.550),
        policy("ppo", 0.521),
        policy("oracle-lookahead", 0.566),
        policy("blind-lookahead", 0.281),
    ]


def test_simulation_question_is_answered_from_stored_results_and_marked_simulated():
    core = TerraCore(Retriever(sim_corpus(), embedder=HashingEmbedder()))
    report = core.ask("Which response policy works best in the simulation?")
    assert report["verdict"] == "answer"
    # The oracle is an upper reference, so the best fair policy comes first.
    assert report["evidence"][0]["id"] == "simulation:e7:test:lookahead"
    assert "0.550" in report["answer"] and "0.521" in report["answer"]
    assert any("Simulated outcome" in c for c in report["caveats"])


def test_simulation_question_abstains_when_no_results_are_stored(core):
    report = core.ask("Which response policy works best in the simulation?")
    assert report["verdict"] == "abstain"
