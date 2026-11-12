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

**Phase 8: validation and paper package.**

- `terramind validate` regenerates `docs/results.md`, `experiments/results.json`
  and four figures from run files, and evaluates each research question's
  verdict as a rule over the numbers. Verdicts: RQ1 not supported; RQ2
  supported, weakly tested; RQ3 supported at pixel level; RQ4 supported for the
  hybrid score, not for the graph; RQ5 not tested; RQ6 not supported; RQ7
  partly (perception matters, uncertainty and calibration barely do).
- **New finding: event-level intervals are far wider than seed noise.**
  Resampling the 10 test events gives test IoU intervals of [0.498, 0.749] for
  radar, [0.764, 0.890] for optical and [0.765, 0.875] for fusion, against seed
  std below 0.01. Earlier entries that read small differences off seed std
  (for example optical against fusion on test, 0.845 against 0.834) overstate
  what is known. Optical against radar still holds. The intervals are not
  paired, so they are conservative for comparisons between inputs.
- Protocol checks: 7 of 8 pass. The failing one is structural: the unseen
  region is a single event, so no event-level interval can be formed for it.
- Literature and licence check (2026-11-11, search results only; nothing read):
  Sen1Floods11, Prithvi-EO-2.0, PANGAEA, GEO-Bench and Spatial-RAG exist with
  the references now in the reading list; three citations remain unchecked.
  The Sen1Floods11 repository states no licence, so the data must not be
  redistributed. How its labels were drawn is still unverified.
- Two of those papers cut against novelty claims: Spatial-RAG already describes
  a hybrid spatial and semantic retriever, and PANGAEA's listing says foundation
  models do not consistently beat a supervised U-Net.
- Paper draft (`paper/draft.md`) reports the negative results as the main
  finding; related work is left as TODO because the papers have not been read.
  Demo script: `docs/demo.md`.

**E9: a harder simulated task, where mistakes cost something.**

- Why: E7/E8 could not separate policies because a wrong guess cost one step.
- Task (`src/terramind/sim/dispatch.py`, `terramind dispatch-eval`): moving
  and deploying are separate actions. A deployment takes a step and serves the
  cell; if the cell is dry (under 5% water) it is wasted at a cost of
  0.25, 0.5 or 1 fully flooded cells. Nothing is revealed until a team deploys,
  then only the 3 x 3 cells around it. Utility = (flooded area served - cost x
  wasted deployments) / total flooded area. Still invented; no exposure data.
- Planner: deploy where expected water minus cost x P(dry) is positive. What
  changes is P(dry): the map at face value; a logistic model of the prediction
  fitted on validation cells ("calibrated"); the same with ensemble
  disagreement as extra inputs.
- Dry-cell prediction, Brier score: face value 0.152 / calibrated 0.084 / with
  uncertainty 0.078 on test; 0.113 / 0.085 / 0.084 on Bolivia.

Utility, with the paired difference to the calibrated planner:

| Waste cost | Planner uses | Test (44) | Bolivia (11) |
|---|---|---|---|
| 0.25 | Map at face value | 0.597 (-0.013 [-0.033, +0.004]) | 0.592 (+0.015 [-0.001, +0.034]) |
| 0.25 | Calibrated | 0.610 | 0.576 |
| 0.25 | Calibrated + uncertainty | 0.610 (+0.000 [-0.012, +0.012]) | 0.573 (-0.003 [-0.030, +0.019]) |
| 0.25 | Optical, calibrated | 0.636 (+0.026 [-0.002, +0.065]) | 0.640 (+0.064 [-0.004, +0.160]) |
| 0.25 | Truth | 0.657 (+0.047) | 0.662 (+0.086) |
| 0.5 | Map at face value | 0.557 (-0.039 [-0.086, -0.000]) | 0.545 (+0.034 [-0.015, +0.088]) |
| 0.5 | Calibrated | 0.596 | 0.511 |
| 0.5 | Calibrated + uncertainty | 0.598 (+0.002 [-0.007, +0.014]) | 0.519 (+0.008 [-0.021, +0.049]) |
| 0.5 | Optical, calibrated | 0.624 (+0.028 [-0.007, +0.073]) | 0.628 (+0.117 [+0.015, +0.258]) |
| 0.5 | Truth | 0.657 (+0.061) | 0.662 (+0.151) |
| 1.0 | Map at face value | 0.476 (-0.090 [-0.184, -0.017]) | 0.451 (+0.021 [-0.053, +0.103]) |
| 1.0 | Calibrated | 0.566 | 0.430 |
| 1.0 | Calibrated + uncertainty | 0.587 (+0.021 [+0.002, +0.044]) | 0.430 (-0.001 [-0.040, +0.036]) |
| 1.0 | Optical, calibrated | 0.585 (+0.019 [-0.041, +0.082]) | 0.616 (+0.186 [+0.031, +0.422]) |
| 1.0 | Truth | 0.657 (+0.091) | 0.662 (+0.232) |

- **The task now separates methods.** The gap between radar and the truth is
  0.05-0.09 on test and 0.09-0.23 on Bolivia, against 0.02-0.04 before.
- **Calibration helps in distribution and grows with the stakes:** on test it
  is worth 0.013, 0.039 and 0.090 as the cost rises, with intervals excluding
  zero at 0.5 and 1.0. Wasted deployments fall from 1.7 to 0.4-0.9 per episode.
- **Calibration does not transfer.** On Bolivia the face-value map scores
  higher at every cost (not significantly, 11 scenarios). A calibration fitted
  on seen regions is no better than none on the unseen one. This is the same
  effect seen at pixel level (temperature scaling half-fixed Bolivia ECE).
- **Uncertainty helps once.** Adding ensemble disagreement gains 0.021
  [0.002, 0.044] at the highest cost on test and nothing elsewhere. First
  positive evidence for the thesis, and weak: 1 of 6 cases, and it would not
  survive a correction for six comparisons.
- **A better sensor is worth more than either** on Bolivia: optical gains
  0.06-0.19 there.
- Limits: 11 Bolivia scenarios; one planner; the cost structure and the 5% dry
  threshold are invented; a wasted deployment is costed relative to each
  scenario's total water, so small floods dominate the variance.
**PPO on the dispatch task** (waste cost 0.5; 3 seeds each; 300 updates x 64
episodes; team-id input added).

| Policy | Test utility | vs calibrated planner | Bolivia utility | vs calibrated planner |
|---|---|---|---|---|
| Calibrated planner | 0.596 | | 0.511 | |
| PPO | 0.404 | -0.192 [-0.254, -0.132] | 0.327 | -0.184 [-0.258, -0.105] |
| PPO, no uncertainty input | 0.424 | -0.172 [-0.231, -0.116] | 0.337 | -0.174 [-0.250, -0.095] |

- **RQ6 again not supported, by a wider margin.** PPO is about 0.19 below the
  planner on both splits and below even the face-value planner (0.557).
- It makes fewer deployments (32 against 40) and wastes more of them (1.3
  against 0.6 on test, 3.0 against 1.8 on Bolivia).
- The uncertainty input does not help PPO; the variant without it is slightly
  ahead, within noise.
- Training plateaued rather than ran out of time: seed 0's training utility
  was 0.45 at update 150 and 0.45 at update 300. It also drops from 0.46 on
  its training scenarios to 0.40 on test, so it partly memorises 45 scenarios.
- The team-id input fixed the symmetry problem: taking the most likely move
  now scores 0.37 on test against 0.40 for sampled moves, where before it
  halved the score.
- Limits: one hyperparameter setting, no tuning, a small network, 45 training
  scenarios. This says PPO as configured loses, not that RL cannot do this.

**RQ5: agents against a fixed pipeline and one LLM agent** (`terramind
agents-eval`, `src/terramind/agents/evaluate.py`).

- 49 frozen questions (fingerprint `6e690dbc3f0c`): 15 on model performance
  by event, 5 on leave-one-event-out transfer, 6 scene counts, 6 licences,
  5 simulation results, 12 that the store cannot answer (forecasts, unknown
  places, deaths, rainfall, damage, evacuation, exposure, funding). Gold facts
  come from the evidence store. The wording was fixed before any system ran.
- Systems, all on the same hybrid retriever: TerraCore; TerraCore with a critic
  that accepts everything; a fixed pipeline that answers with the top record
  and never refuses; llama3:8b given the top 8 records and told to cite ids or
  reply INSUFFICIENT EVIDENCE.

| System | Accuracy | Answerable correct | Unanswerable refused | Answerable refused | Cites a supporting record | Decimals not in cited records |
|---|---|---|---|---|---|---|
| TerraCore | 0.78 | 0.89 | 0.42 | 0.03 | 0.94 | 0.007 |
| TerraCore, no critic | 0.78 | 0.86 | 0.50 | 0.05 | 0.94 | 0.007 |
| Fixed pipeline | 0.43 | 0.57 | 0.00 | 0.00 | 0.68 | 0.000 |
| LLM agent (llama3:8b) | 0.67 | 0.57 | 1.00 | 0.14 | 0.81 | 0.093 |

Paired accuracy differences, bootstrap over questions: TerraCore - fixed
pipeline +0.35 [+0.20, +0.49]; TerraCore - LLM agent +0.10 [-0.10, +0.31];
TerraCore - no critic 0.00 [-0.06, +0.06].

- **RQ5 supported against the fixed pipeline only.** TerraCore is not
  distinguishable from one LLM agent on 49 questions.
- **The two fail in opposite directions.** TerraCore is right on 89% of
  answerable questions but refuses only 5 of 12 unanswerable ones; the LLM
  refuses all 12 but gets 57% of answerable ones and refuses 5 it could answer.
- **The critic checks faithfulness, not relevance.** Asked about deaths,
  rainfall or damage in a known flood, or about model accuracy in Jakarta or
  Bangladesh, TerraCore returns true, correctly cited benchmark figures that
  do not answer the question, and the critic passes them. This is the main
  failure, and it is a design gap rather than a bug.
- **The critic adds nothing here.** Its relax-and-retry fixed one scene count
  and broke one refusal: asked for Mumbai scenes in 2023, the retry dropped
  the date and, with no place resolved, counted all 154 Sentinel-2 scenes.
- TerraCore also reports the best simulated policy when asked about a named
  one (greedy, plain lookahead), and misreads "November-December 2015".
- The LLM picks the wrong one of two similar records (held-out instead of
  standard model) and cannot count scenes, since counts need more records
  than its top 8. One answer quoted four chip scores without citing them.
- The fixed pipeline fails whenever the top record is a near miss.
- Limits: I wrote both TerraCore's rules and the questions, which favours
  TerraCore; gold facts come from the same store the systems read; one small
  local LLM; 49 questions. The LLM's answers vary slightly between runs (GPU
  inference is not fully deterministic); two runs gave the same accuracy.
- Not done, on purpose: fixing TerraCore's relevance check and rerunning on
  these questions would tune to the test set. A fix needs a new question set.

**2026-11-12: held-out question set frozen before any fix.** The 49 questions
above become the development set ("dev", `6e690dbc3f0c`). A second set of 49
("test", fingerprint `3281dec01689`) was written and frozen before TerraCore
was changed: different wording and targets, no shared question text, two new
categories (calibration, event facts) and 16 new unanswerable questions
(fatalities, rainfall, insured losses, displacement, unknown places, a
forecast, a real allocation, hospitals, water depth, elevation, damage, who
labelled the data, river discharge). Protocol: develop the fix on dev only,
then run every system once on test and report the result whatever it is.
Caveat: the same person wrote the fix and both sets.

**Relevance fixes (developed on dev only).** With `relevance=True` TerraCore
refuses questions naming something absent from the whole store (a capitalised
word not in any record), gives no default answer to a question whose type it
does not recognise, drops dates on retry only when a place still narrows the
search, reports the simulated policy and waste cost the question names, looks
datasets up by name, refuses dataset attributes the store does not record, and
parses month ranges. `relevance=False` reproduces the first version. On dev:
0.80 -> 1.00 (expected; the fixes were built on it). Two of the fixes (waste
cost, dataset names) touch question types I knew the held-out set contained.

**RQ5 on the held-out set** (`terramind agents-eval --split test`, run once):

| System | Accuracy | Answerable correct | Unanswerable refused |
|---|---|---|---|
| TerraCore (fixed) | 0.88 | 0.91 | 0.81 |
| TerraCore first version | 0.59 | 0.82 | 0.12 |
| TerraCore, no critic | 0.88 | 0.91 | 0.81 |
| Fixed pipeline | 0.43 | 0.64 | 0.00 |
| LLM agent (llama3:8b) | 0.78 | 0.67 | 1.00 |

Paired: fixed - first version +0.29 [+0.16, +0.41]; fixed - LLM agent +0.10
[-0.04, +0.24]; fixed - pipeline +0.45 [+0.31, +0.59]; critic adds 0.00.

- The relevance fixes generalise: held-out refusals rose from 2 of 16 to 13 of
  16 and nothing answerable was lost.
- Still failing: all three calibration questions (the agents never look up
  calibration records for a named place; not fixed because no dev question
  needed it), "Predict next month's flood risk" ("predict" is not on the
  forecast list), "Should rescue teams go to the Ahr valley or Sindh first?"
  (routed to the simulation), and "average elevation of the Bolivia chips"
  (answered with chip statistics).
- TerraCore is still not distinguishable from one small LLM agent, and the
  critic still changes nothing: every claim the agents build is faithful by
  construction, so the critic never has anything to reject on real questions.

**E2: foundation model (Prithvi-EO-2.0-100M-TL).** Encoder from IBM/NASA's
Apache-2.0 checkpoint (strict weight load), UNETR-style decoder, same trainer,
crops, schedule and splits as the U-Nets; encoder learning rate 0.05 x decoder.
Six bands (B2, B3, B4, B8A, B11, B12) normalised with Prithvi's statistics.
3 seeds each, water IoU:

| Model | valid | test | Bolivia |
|---|---|---|---|
| U-Net, 13 optical bands (5 seeds) | 0.820 ± 0.011 | 0.845 ± 0.006 | 0.802 ± 0.006 |
| U-Net, the same 6 bands | 0.818 ± 0.006 | 0.839 ± 0.007 | 0.782 ± 0.005 |
| Prithvi, fine-tuned | 0.805 ± 0.002 | 0.817 ± 0.003 | 0.691 ± 0.022 |
| Prithvi, frozen encoder | 0.786 ± 0.004 | 0.801 ± 0.002 | 0.662 ± 0.014 |

- **Negative result:** the foundation model is worse than a small U-Net trained
  from scratch on the same bands, by 0.02 on test and 0.09 on unseen Bolivia.
  It transfers worse, not better. This agrees with PANGAEA's reported finding
  that foundation models do not consistently beat a supervised U-Net.
- Band choice explains little: six bands instead of thirteen costs the U-Net
  0.006 on test and 0.02 on Bolivia.
- Limits: one recipe and no tuning; Prithvi was pretrained on 30 m HLS and is
  applied to 10 m Sentinel-2 without its time/location inputs; patch size 16
  limits boundary detail; 3 seeds. The 300M and 600M models were not tried.

**E3: discharge forecasting** (`terramind forecast-fetch`, `terramind
forecast`). Target: GloFAS v4 reanalysis discharge at the largest river cell
near each of the 15 event sites (searched within about 50 km where the nearest
cells were trickles). Inputs: the last 30 days of discharge and ERA5
precipitation, no future rain. Train 1991-2012, test 2017-2024; one model pooled
over sites. Event dates: Sen1Floods11's own metadata file for 11 events,
configs/events.yaml (unconfirmed) for 4.

| Lead | Persistence NSE | Ridge | Gradient boosting | GRU | GRU error vs persistence |
|---|---|---|---|---|---|
| 1 day | 0.985 | 0.988 | 0.985 | 0.986 | -27% |
| 3 days | 0.941 | 0.924 | 0.909 | 0.914 | -17% |
| 7 days | 0.823 | 0.809 | 0.798 | 0.829 | -16% |

(median NSE over sites; the last column is squared error on standardised log
discharge.) Flood-threshold exceedance (Q > training 95th percentile), critical
success index: 1 day GRU 0.83 vs persistence 0.78; 3 days boosting 0.58 vs 0.55;
7 days persistence best (0.39).

- Persistence is very hard to beat on a large slow river; only the GRU improves
  on it at every lead, and only modestly.
- **The model river misses real floods:** GloFAS exceeds its threshold within a
  week of the satellite-observed flood at 10 of 13 test-period events; at
  Paraguay, Somalia and Sri Lanka it does not, so no forecast of it could warn.
- Flash floods are not forecastable this way: Kerala and Ahr are caught 1 day
  ahead at best.
- Pooling transfers: leave-one-site-out NSE at 3 days is within 0.05 of the
  pooled model at every site.
- Limits: the target is a simulation, not gauges; one river cell per site may
  not be the flooded river; no ensemble weather forecasts as input.

**Storage, explorer and report.**

- `src/terramind/storage`: a spatial database (PostGIS, or embedded DuckDB with
  the spatial extension) holding chips, scenes and evidence with geometry
  indexes; an evidence graph loader and one-hop query for Neo4j; an object store
  (MinIO, or local disk) for map layers and run reports. CLI `db-load`,
  `graph-load`, `publish-objects`; the API uses the spatial database for bounding
  boxes when `TERRAMIND_SPATIAL=duckdb` or `TERRAMIND_DATABASE_URL` is set.
  docker-compose now has PostGIS, MinIO and Neo4j, with generated passwords in a
  gitignored `.env`.
- **Verified:** DuckDB loads 446 chips, 223 scenes and 928 evidence records and
  returns the same bounding-box answers as the in-memory path (15 Bolivia chips,
  18 Kerala Sentinel-2 scenes); 2,676 layers and 66 reports publish to the local
  object store. **Not verified:** PostGIS, Neo4j and MinIO. Docker Desktop's
  backend crashes on start because a dead socket file
  (`%LOCALAPPDATA%\Docker\run\dockerInference`) cannot be removed without
  administrator rights.
- Explorer: a 2D map / 3D globe toggle (MapLibre globe projection, terrain and
  hillshade from the AWS terrain tiles). Cesium was not used: MapLibre already
  renders a globe with terrain and needs no access token.
- LLM-written summary of the situation report (`terramind report --llm`, API
  `?narrate=true`, explorer button): llama3:8b writes it; it is shown only if it
  adds no number absent from the verified statements. The first attempt was
  rejected for rounding 70.2% to 70: the guard is strict by design.

**Licences and citations (2026-11-12).** 13 of 17 registry entries verified on
the provider's own page or licence document (`licence_source` in
configs/datasets.yaml). Changes from what was recorded: ERA5 is CC BY 4.0 since
2 July 2025; the Copernicus DEM licence requires a specific notice and a
liability disclaimer when redistributed; WorldPop products derived from OSM or
Microsoft data are ODbL; GHSL asks for a specific paper citation. Not verified:
geoBoundaries (licence varies per file), SpaceNet ("Various"), xBD (terms not
readable; secondary sources say CC BY-NC-SA 4.0), terrain tiles. Sen1Floods11:
confirmed that no licence is given. All eight cited works exist (none read in
full); GeoAgentic-RAG (2026) is a multi-agent geospatial RAG, the closest prior
work to TerraCore.

**Open items.**

- Fix Docker (delete the dead socket file as administrator, or reboot), then
  verify PostGIS, Neo4j and MinIO.
- Confirm event boxes and dates for the four candidate events.
- More seeds for leave-one-event-out and a size-matched control.
- Read the cited papers in full.
