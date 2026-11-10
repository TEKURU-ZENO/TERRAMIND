# Research log

One entry per week: an engineering artefact, an experimental result, and what
was learned. Record negative results.

## 2026-11-10 (Sprint 0, day 1)

**Engineering.** Repository scaffolded as a modular monolith: dataset/licence
registry, flood event catalogue, STAC discovery, Sen1Floods11 downloader,
Sentinel-1 U-Net baseline with per-event evaluation.

**Experiment.** None that counts yet. Any smoke run on a handful of chips only
proves the pipeline executes; it is not a result.

**First full baseline (single seed, so no confidence intervals).**
Run `unet-s1-20261110T070513Z`: U-Net, random init, Sentinel-1 VV+VH only,
252 training chips, 30 epochs, seed 0, checkpoint chosen on validation IoU
(epoch 24).

| Split | Chips | Water IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|
| valid | 89 | 0.609 | 0.757 | 0.848 | 0.684 |
| test (in-distribution) | 90 | 0.651 | 0.788 | 0.842 | 0.741 |
| Bolivia (unseen region) | 15 | 0.598 | 0.749 | 0.914 | 0.634 |

Per-event test IoU ranges from 0.16 (Pakistan) and 0.38 (Somalia) to 0.88
(Sri-Lanka). The pooled number hides this spread. The model is
precision-heavy: it misses water more than it invents it.

**Five seeds (0-4), same settings.** Mean ± sample std across seeds; this is
training variance only, not uncertainty over events or regions. Reproduce with
`python experiments/baselines/aggregate.py`.

| Split | Water IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| valid | 0.613 ± 0.003 | 0.760 ± 0.002 | 0.811 ± 0.047 | 0.719 ± 0.043 |
| test (in-distribution) | 0.648 ± 0.009 | 0.786 ± 0.007 | 0.812 ± 0.041 | 0.764 ± 0.027 |
| Bolivia (unseen region) | 0.628 ± 0.045 | 0.771 ± 0.034 | 0.896 ± 0.023 | 0.680 ± 0.066 |

- Bolivia IoU ranges 0.57-0.68 across seeds, five times the test spread. With
  this variance the Bolivia-vs-test gap (0.02) is not distinguishable from zero.
- Weak events are stable across seeds, so they are real failure modes, not
  noise: test Pakistan 0.18 ± 0.02, valid Somalia 0.14 ± 0.01.
- IoU is stable while precision and recall trade off between seeds (std ~0.04).

**Non-learned baselines** (`experiments/baselines/threshold_baseline.py`).
Water = VH backscatter below a threshold.

| Method | valid IoU | test IoU | Bolivia IoU |
|---|---|---|---|
| Otsu per chip | 0.220 | 0.232 | 0.380 |
| One global threshold (tuned on train) | 0.493 | 0.533 | 0.535 |
| U-Net S1, mean of 5 seeds | 0.613 | 0.648 | 0.628 |

The U-Net adds about 0.10-0.12 IoU over a single tuned threshold. Per-chip
Otsu over-predicts water (precision ~0.25) because it always splits a chip in
two, including chips with no water.

**Uncertainty: 5-seed deep ensemble, S1** (`terramind eval-ensemble`).
Pixel-level; temperature fitted on valid only.

| Split | Model | IoU | ECE | NLL | Error at 100 / 90 / 80% coverage |
|---|---|---|---|---|---|
| test | single (mean) | 0.648 | 0.021 | 0.153 | 5.1% / 2.4% / 1.4% |
| test | ensemble | 0.657 | 0.024 | 0.149 | 4.9% / 2.3% / 1.4% |
| test | ensemble + temperature | 0.657 | 0.005 | 0.143 | same ranking |
| Bolivia | single (mean) | 0.628 | 0.019 | 0.163 | 6.4% / 3.4% / 1.6% |
| Bolivia | ensemble | 0.640 | 0.022 | 0.158 | 6.2% / 3.3% / 1.6% |
| Bolivia | ensemble + temperature | 0.640 | 0.012 | 0.156 | same ranking |

- The ensemble gains ~0.01 IoU and slightly better NLL, but its raw ECE is
  *worse* than a single model. The members are under-confident (fitted
  temperature 0.74-0.77 < 1) and averaging makes that worse.
- Temperature scaling fixes in-distribution calibration (test ECE 0.024 ->
  0.005) but only half-fixes it on Bolivia (0.022 -> 0.012). Calibration learned
  on seen regions does not fully transfer: this is the RQ3 effect to study.
- Abstention works: dropping the least confident 10% of pixels halves the error
  rate on every split.
- Caveat: pixel error rate is dominated by easy land pixels. Selective IoU and
  chip-level abstention are still to do.

**E1 (partial): modality comparison.** Same U-Net, 5 seeds each, scored on
pixels valid in both sensors. Early fusion = 15 input channels.

| Input | valid IoU | test IoU | Bolivia IoU |
|---|---|---|---|
| S1 radar (2 ch) | 0.613 ± 0.003 | 0.648 ± 0.009 | 0.628 ± 0.045 |
| S2 optical (13 ch) | 0.820 ± 0.011 | 0.845 ± 0.006 | 0.802 ± 0.006 |
| S1+S2 early fusion | 0.816 ± 0.004 | 0.834 ± 0.008 | 0.757 ± 0.024 |

- **Negative result for RQ1 as naively posed:** early fusion does not beat
  optical alone, and on unseen Bolivia it is clearly worse (0.757 vs 0.802,
  seed ranges do not overlap). Adding radar hurt transfer.
- Optical beats radar by ~0.20 IoU. Two caveats before reading this as "optical
  is better": (1) the hand labels are believed to have been drawn with the
  optical imagery in view, which favours optical models (to verify against the
  dataset paper); (2) cloudy pixels are excluded from the labels, so the
  benchmark does not test the case radar exists for.
- Next for fusion: modality dropout, late fusion, and a radar-only evaluation
  of the fused model (optical zeroed) to measure reliance on optical.

**Ensemble calibration by modality** (pixel level, common mask, temperature
fitted on valid).

| Input | test ECE raw -> scaled | Bolivia ECE raw -> scaled | Bolivia error at 100 / 90% coverage |
|---|---|---|---|
| S1 | 0.024 -> 0.005 | 0.022 -> 0.012 | 6.2% / 3.3% |
| S2 | 0.018 -> 0.005 | 0.037 -> 0.015 | 3.3% / 0.9% |
| S1+S2 | 0.016 -> 0.003 | 0.019 -> 0.005 | 4.0% / 1.0% |

- All models are under-confident (temperatures 0.67-0.77).
- The optical model is the most accurate on Bolivia but the worst calibrated
  there; the fused model is less accurate but its calibration transfers better.
  Accuracy and calibration under shift do not move together here.

**Phase 2: Earth Explorer** built (`apps/web`, `src/terramind/api`): chip map,
overlays for label / prediction / errors / ensemble disagreement, scene
footprints for candidate events. 2D only; no Cesium globe.

**E4: leave-one-event-out transfer.** 33 runs: each of 11 events held out, for
S1, S2 and S1+S2, seed 0, 30 epochs. "Seen" = 5-seed mean of the standard model
on that event's test chips; "unseen" = the held-out model on the same chips.
Reproduce with `python experiments/transfer/loeo_summary.py --modalities s1 s2 s1s2`.

| Input | Mean IoU drop when unseen (10 events) | Worst drop | Events that got worse |
|---|---|---|---|
| S1 | 0.002 | 0.019 (Paraguay) | 6 of 10 |
| S2 | 0.010 | 0.043 (Pakistan), 0.041 (Somalia) | 8 of 10 |
| S1+S2 | 0.007 | 0.094 (Ghana) | 6 of 10 |

- Holding an event out of training barely changes the score on it. The drops
  are the size of seed noise (test std 0.006-0.009 pooled, larger per event).
- So the weak events are hard in themselves, not under-represented: S1 Pakistan
  is 0.18 seen and 0.18 unseen; S1 Somalia 0.45 and 0.47.
- Bolivia, never seen by either model, scores 0.01-0.03 higher with the
  held-out models, which train on ~40% more chips (train + test splits of the
  other events). That is the size of the extra-data effect, and it biases every
  "unseen" number upward, so the true cost of not seeing an event is somewhat
  larger than the table shows.
- The one large drop, fusion on Ghana (0.611 -> 0.518), is a single seed on a
  handful of test chips; not to be read as a finding without more seeds.
- Limits: one seed per fold; events on the same continent stay in training, so
  this is leave-one-event-out, not leave-one-region-out; chip counts per event
  are small.

**Phase 5: evidence retrieval and agents.**

- Evidence corpus (`terramind build-evidence`): 782 records built only from
  files in the repo (446 chips, 223 scenes, 64 benchmark results, 15 events,
  14 datasets, 20 document sections), each with source, place, time and a
  provenance score. Provenance scores are hand-set placeholders.
- Hybrid retrieval (`src/terramind/rag`): semantic (MiniLM via Qdrant local
  mode) + spatial (Shapely) + temporal + provenance, with one-hop graph
  expansion (NetworkX). PostGIS and Neo4j are not used yet; Docker was not
  running.
- TerraCore (`src/terramind/agents`, LangGraph): orchestrator -> evidence ->
  earth observation / GIS / prediction -> critic -> report. No language model
  writes answers. The critic checks every claim's evidence ids, counts, numbers
  and means against the store; it retries once without the time window, else
  abstains. An optional Ollama rewrite is accepted only if it adds no numbers.

**E5 (first version): retrieval benchmark**, 36 generated queries, k = 10
(`terramind rag-benchmark`).

| Method | P@10 | Recall@10 | nDCG@10 | Located results in asked region |
|---|---|---|---|---|
| Vector only | 0.50 | 0.76 | 0.67 | 0.75 |
| Spatial/temporal filter only | 0.36 | 0.61 | 0.53 | 1.00 |
| Hybrid, no graph | 0.55 | 0.89 | 0.79 | 0.99 |
| Hybrid + graph | 0.55 | 0.89 | 0.79 | 0.99 |

- Hybrid beats either alone; the spatial filter alone wins on "performance in
  region X" (nDCG 0.91 vs 0.59 hybrid), so the fixed weights are not right for
  every intent.
- Graph expansion changed nothing on this benchmark. No evidence yet that the
  graph component earns its place.
- The benchmark is synthetic: queries are generated from the corpus and use
  place names the parser already knows. It is a regression test and ablation,
  not evidence of real-world retrieval quality. Weights are uncalibrated.
- Spec target "100% of final claims linked to evidence" holds by construction
  for the templated reports; it is enforced by the critic, not measured on
  free text.

**Phase 6: TerraSim and E7 (all results in this section are SIMULATED).**

- Environment (`src/terramind/sim`): each chip becomes a 16 x 16 grid. Three
  teams start at one edge cell and move one cell per step for 48 steps;
  entering a cell serves it. Per cell, truth = labelled water share, and the
  teams see the S1 ensemble's mean probability and member disagreement, both
  over labelled pixels only. Truth is revealed within one cell of a team.
  Score ("auc") = mean over the episode of the share of flooded area served, so
  early service counts more. Gymnasium wrapper included.
- What is invented: the teams, their movement, and "need = flooded area". No
  population, roads or elevation are used, and labels include permanent water.
- Scenarios: chips with at least 2% water. valid 45 (PPO training and tuning),
  test 44, Bolivia 11; 4 fixed depots each, the same for every policy.
  Intervals are 95% bootstrap over scenarios.
- PPO: one actor shared by the teams, agent-centred views, team reward, critic
  sees the truth during training; 400 updates x 64 episodes, 3 seeds.

| Policy | test auc | Bolivia auc |
|---|---|---|
| Random walk | 0.108 | 0.107 |
| Lookahead ignoring the model | 0.281 | 0.314 |
| Greedy | 0.519 | 0.516 |
| Assignment (Hungarian) | 0.529 | 0.506 |
| Lookahead (5 moves, replanned each step) | 0.550 | 0.550 |
| Lookahead + uncertainty bonus (weight tuned on valid) | 0.549 | 0.555 |
| PPO, 3 seeds (seed std 0.003 / 0.016) | 0.521 | 0.526 |
| PPO without the uncertainty input, 3 seeds | 0.515 | 0.527 |
| Oracle lookahead (sees truth; not the optimum) | 0.566 | 0.588 |

Paired differences in auc (same episodes):

| Comparison | test | Bolivia |
|---|---|---|
| PPO - lookahead | -0.029 [-0.039, -0.019] | -0.024 [-0.041, -0.007] |
| PPO - assignment | -0.009 [-0.016, -0.002] | +0.020 [-0.016, +0.071] |
| PPO - PPO without uncertainty | +0.005 [+0.002, +0.009] | -0.001 [-0.007, +0.004] |
| Lookahead + uncertainty - lookahead | -0.001 [-0.004, +0.002] | +0.005 [-0.003, +0.015] |
| Lookahead - lookahead ignoring the model | +0.268 [+0.233, +0.303] | +0.236 [+0.161, +0.313] |
| Oracle - lookahead | +0.017 [+0.008, +0.029] | +0.038 [+0.013, +0.072] |

- **RQ6, negative result:** PPO does not beat the planning baseline. It is
  about level with greedy and assignment and 0.02-0.03 below lookahead on both
  splits. It does transfer: no drop from test to unseen Bolivia.
- **Uncertainty did not help decisions here.** The bonus changes lookahead by
  less than 0.005 either way, and PPO with the uncertainty input gains 0.005 on
  test and nothing on Bolivia. This is evidence against the project's thesis
  in this environment, not for it.
- **Perception quality hardly matters in this task.** Using the model is worth
  +0.27 over ignoring it, but the gap to seeing the truth is only 0.02-0.04,
  and swapping radar for optical perception (0.20 higher IoU) moves lookahead
  from 0.550 to 0.558 on test (0.550 to 0.577 on Bolivia, 11 scenarios). Cell
  averages and reveal-on-approach forgive most pixel errors.
- **The environment is too easy to separate policies:** lookahead already
  serves 90% of the flooded area. Before more RL work it needs to be harder and
  more consequential: fewer teams or steps, travel cost that depends on
  flooding, real exposure (population, roads), and a forecast instead of a
  snapshot.
- **Symmetry bug found and documented:** taking PPO's most likely move scores
  0.290, because teams share one actor and start with identical views, so they
  move as one. Moves are therefore sampled (seeded) at evaluation, as in
  training. An agent-id input would be the proper fix; not done.
- Limits: PPO trained on only 45 scenarios (augmented by rotation/reflection)
  with a fixed budget and no tuning, so "PPO loses" is specific to this setup;
  beta and PPO both used the valid split; Bolivia is 11 scenarios.

**Phase 7: integration and E8 (simulated).**

- Simulation results are evidence records (kind `simulation`, provenance 0.7,
  hand-set). TerraCore has a decision agent that answers simulation questions
  only from those records and labels them simulated; forecasting still abstains.
- Situation report (`terramind report <chip>`, `/api/report/<chip>`, chip card
  in the explorer): perception and disagreement computed from the ensemble,
  reliability from the event's benchmark evidence, then a simulated response.
  Every statement carries evidence ids or "computed". It recommends no
  allocation when the event's radar IoU is below 0.40 (hand-set).

**E8: chain ablation** (`terramind chain-ablation`). The planner is fixed
(lookahead); only what it is given changes. auc on test, with the paired
difference to the radar ensemble and its 95% interval:

| Planner is given | standard (3 teams, 48 moves) | scarce (2 teams, 24 moves) | scarce, truth revealed only on entry |
|---|---|---|---|
| Nothing | 0.281 (-0.268) | 0.106 (-0.129) | 0.106 (-0.123) |
| Radar ensemble, thresholded | 0.540 (-0.009 [-0.018, -0.002]) | 0.234 (-0.002) | 0.225 (-0.005) |
| Radar, one model | 0.546 (-0.004 [-0.007, -0.001]) | 0.234 (-0.002) | 0.229 (-0.001) |
| Radar ensemble (reference) | 0.550 | 0.236 | 0.230 |
| Radar ensemble, temperature-scaled | 0.550 (+0.000) | 0.237 (+0.001) | 0.232 (+0.002 [+0.001, +0.004]) |
| Radar ensemble + uncertainty weight | 0.549 (-0.001) | 0.235 (-0.000) | 0.232 (+0.002 [+0.000, +0.005]) |
| Optical ensemble | 0.558 (+0.008 [-0.000, +0.020]) | 0.246 (+0.010 [+0.003, +0.020]) | 0.241 (+0.011 [+0.003, +0.023]) |
| Fused ensemble | 0.558 (+0.008 [+0.001, +0.018]) | 0.248 (+0.012 [+0.006, +0.022]) | 0.240 (+0.011 [+0.003, +0.021]) |
| Truth (oracle) | 0.566 (+0.017) | 0.253 (+0.017) | 0.253 (+0.023) |

- **RQ7, mostly negative:** having a flood map at all is worth 0.12-0.27. Every
  refinement after that (probabilities instead of a thresholded map, ensemble
  instead of one model, calibration, uncertainty) is worth 0.01 or less, and
  several intervals include zero. The whole gap from radar to truth is
  0.02-0.04.
- The only refinement that reliably helps is a better sensor: optical or fused
  input gains about 0.01 on test in every setting, closing roughly half the gap
  to the oracle. Fusion, which lost on IoU, is level with optical here.
- Making the task harder (fewer teams and moves, no reveal) did not make
  uncertainty or calibration matter more. On Bolivia (11 scenarios) the signs
  flip between settings; nothing there is distinguishable from zero.
- The uncertainty weight chosen on valid is +1, +2 and -1 across the three
  settings, i.e. no consistent direction.
- Reading: in this problem need is spatially smooth and aggregated to cells,
  so a coarse map is enough. The thesis that uncertainty propagation improves
  decisions is not supported here; testing it fairly needs a task where
  errors are costly (false alarms that waste scarce teams, travel through
  flooded cells, need defined by exposure).
- Not built in Phase 7: PostGIS/MinIO/Neo4j, Cesium, an LLM-written report,
  forecasting.

**Open items.**

- Verify every licence in `configs/datasets.yaml`.
- Verify the citations in `research/literature/reading-list.md`.
- Confirm event boxes and dates in `configs/events.yaml`.
- More seeds for the leave-one-event-out folds; a size-matched control (same
  number of training chips) to separate "unseen" from "more data".
- Fusion: modality dropout, late fusion, radar-only evaluation of the fused model.
