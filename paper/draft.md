# From flood maps to decisions: does propagating uncertainty and evidence help?

Working draft. Every number comes from `docs/results.md`, which
`terramind validate` regenerates from run files. Simulation results are
simulated outcomes. Sections marked TODO are not written because the work
behind them has not been done.

## Abstract

We built an end-to-end chain for flood intelligence on the Sen1Floods11
benchmark: flood mapping from radar, optical and fused imagery; ensemble
uncertainty and calibration; retrieval of evidence with provenance; a
rule-checked agent workflow; and a simulated response problem with planning
and reinforcement-learning policies. We then asked whether the links add up.
Mostly they do not. Optical input beats radar by about 0.20 IoU and early
fusion does not beat optical. Holding a flood event out of training changes
its score by 0.01 or less, so the hard events are hard in themselves.
Uncertainty is useful for abstention at pixel level, but in simulation neither
uncertainty nor calibration improves decisions measurably: a planner with any
flood map reaches within 0.02-0.04 of one that sees the truth, and a learned
policy does not beat a five-step lookahead. When we make mistakes costly, by
charging for deployments to dry cells, calibration starts to matter on the
test events (up to +0.09 utility) but not on the unseen region, uncertainty
helps in one of six cases, and a better sensor helps most; a learned policy
trained from scratch falls 0.19 behind the planner, and one that first imitates
the planner comes within noise of it without beating it. On 49 held-out
questions, the evidence-checked agent workflow beats a fixed retrieval pipeline
but not a single small LLM agent, and its critic never changes an answer. A
geospatial foundation model (Prithvi-EO-2.0) underperforms a small U-Net,
especially on the unseen region; a fused radar-optical model collapses when the
optical bands are missing; and a graph network over road topology adds nothing
to the flood map for finding flooded roads. Intervals that resample flood events are an
order of magnitude wider than seed noise, which changes how several of these
comparisons should be read. We report the negative results in full.

## 1. Introduction

The starting hypothesis was that uncertainty and evidence should propagate
from perception through simulation to action, and that this coupling improves
decisions under geographic shift. Seven research questions follow from it
(Table 1). This draft reports what a small, fully reproducible implementation
found for each, on one benchmark and one simulated task.

Contributions, stated narrowly:

1. A per-event and leave-one-event-out evaluation of radar, optical and fused
   U-Nets on Sen1Floods11 with event-level intervals.
2. A measurement of how much each perception refinement is worth to a
   downstream (simulated) decision, including the null results.
3. An open pipeline in which every reported number is regenerated from run
   files and every answer or report statement carries evidence ids.

## 2. Related work

TODO. The reading list (`research/literature/reading-list.md`) confirms that
five of the eight works cited in the plan exist; none has been read yet, and
three are unchecked. Two points already bear on our claims and must be
addressed after reading: Spatial-RAG describes a hybrid spatial and semantic
retriever, so our hybrid score is not a contribution; and PANGAEA's listing
reports that foundation models do not consistently beat a supervised U-Net,
which is the comparison we have not run (E2).

## 3. Data

Sen1Floods11 hand-labelled subset: 446 chips of 512 x 512 pixels at 10 m from
11 flood events, with Sentinel-1 VV/VH, Sentinel-2 (13 bands) and labels
(water, not water, no data). Official split: 252 train, 89 validation, 90 test,
and 15 Bolivia chips held out as an unseen region. The repository states no
licence; we do not redistribute any data. Labels mark all surface water, not
only flood water. We have not verified how the labels were drawn; if analysts
used the optical image, optical models are favoured by construction.

## 4. Methods

**Mapping.** A U-Net (base width 32, depth 4, random initialisation) trained
for 30 epochs with AdamW and a one-cycle schedule on 256-pixel crops. Inputs:
radar (2 channels), optical (13) or both stacked (15). The checkpoint is chosen
on validation IoU only. Five seeds per input. Non-learned baselines: per-chip
Otsu and one global VH threshold tuned on train.

**Uncertainty.** The five seeds form an ensemble. We report expected
calibration error before and after temperature scaling fitted on validation,
and the error rate when the least confident pixels are dropped.

**Transfer.** For each of the 11 events and each input, one model is trained
with that event removed and scored on the event's test chips (33 runs, one
seed each).

**Evidence and agents.** An evidence store of 890 records built from run files
and configuration; retrieval scored by semantic, spatial, temporal and
provenance terms; a LangGraph workflow whose critic rejects any claim whose
evidence ids or numbers are not in the store. No language model writes
answers. To test it (RQ5) we froze 49 questions with gold facts from the store:
37 answerable (performance, transfer, scene counts, licences, simulation
results) and 12 the store cannot answer. We compare TerraCore, TerraCore with a
critic that accepts everything, a fixed pipeline that answers with the top
retrieved record, and one LLM agent (llama3:8b, temperature 0) given the top 8
records and told to cite ids or refuse. An answer is correct if it contains
every gold number (to 3 decimals), count or licence name, or, for an
unanswerable question, if it refuses.

**Simulation.** Each chip becomes a 16 x 16 grid. Three teams start at an edge
cell and move one cell per step for 48 steps; entering a cell serves it. Teams
see the ensemble's mean probability and disagreement; truth is revealed within
one cell. The score is the mean over the episode of the share of flooded area
served. Policies: random, greedy, Hungarian assignment, five-step lookahead,
and PPO with a shared actor and a critic that sees the truth during training
(3 seeds, trained on the 45 validation scenarios). In the chain ablation the
planner is fixed and only its input changes.

**Dispatch with costly mistakes.** A second task separates moving from
deploying. A deployment takes a step and serves the cell; if the cell is dry
(under 5% water) it is wasted and costs 0.25, 0.5 or 1 fully flooded cells.
Nothing is revealed until a team deploys, then only the 3 x 3 cells around it.
Utility is flooded area served minus the waste penalty, divided by the total
flooded area. The planner deploys where expected water minus the cost times
the chance of a dry cell is positive, nearest and best first. Only that chance
varies: the map at face value; a logistic model of the prediction fitted on
validation cells; the same with ensemble disagreement as extra inputs; the
optical model, calibrated the same way; and the truth. PPO is retrained for
this task (waste cost 0.5, 3 seeds, with and without the uncertainty input)
with a team identifier in each team's view.

**Statistics.** Seed standard deviations describe training noise. For mapping
we also report 95% intervals from resampling the 10 test events. Simulation
intervals resample scenarios, and policy comparisons are paired on identical
episodes.

## 5. Results

**Table 1. Verdicts.**

| | Question | Verdict |
|---|---|---|
| RQ1 | Does fusion beat the best single sensor? | Not supported |
| RQ2 | Does the model generalise to unseen events? | Supported, weakly tested |
| RQ3 | Can uncertainty identify where to abstain? | Supported at pixel level |
| RQ4 | Does hybrid spatial retrieval help? | Supported for the hybrid score; not for the graph component |
| RQ5 | Are specialised agents more reliable? | Supported against a fixed pipeline; not distinguishable from one LLM agent |
| RQ6 | Does RL beat heuristic and optimisation baselines? | Not supported, in both tasks; imitation + PPO comes within noise |
| RQ7 | Does the integrated chain improve decisions? | Partly: perception matters, uncertainty and calibration barely do |
| RQ7b | When mistakes cost something, do calibration and uncertainty help? | Calibration: in seen regions; uncertainty: rarely; neither in the unseen region |

### 5.1 Mapping (Figure 1)

Test IoU is 0.648 for radar, 0.845 for optical and 0.834 for fusion, against
0.533 for a tuned global threshold. Seed standard deviations are below 0.01,
but intervals over events are wide: [0.498, 0.749] for radar, [0.764, 0.890]
for optical and [0.765, 0.875] for fusion. The spread across events is the
main uncertainty. Radar fails on Pakistan (0.18) and is weak on Somalia, the
USA and Ghana; optical's worst test event is Ghana (0.55). On unseen Bolivia,
fusion (0.757) is below optical (0.802) with non-overlapping seed ranges.

### 5.2 Calibration

All ensembles are under-confident (fitted temperatures 0.67-0.74). Temperature
scaling brings test ECE to 0.005 or below but transfers less well to Bolivia
(radar 0.022 to 0.012). Dropping the least confident 10% of pixels cuts the
Bolivia error rate to 53% of its value for radar and to 25-29% for optical and
fusion.

### 5.3 Transfer (Figure 2)

Holding an event out lowers its IoU by 0.002 (radar), 0.010 (optical) and
0.007 (fusion) on average. The largest single drop is 0.094 (fusion, Ghana),
from one seed. Bolivia scores slightly higher with the held-out models, which
train on about 40% more chips, so these drops understate the true cost
somewhat.

### 5.4 Retrieval

On 36 generated queries, nDCG@10 is 0.67 for vector search, 0.53 for a
spatial-temporal filter and 0.79 for the hybrid score. One-hop graph expansion
changes nothing.

### 5.5 Simulated response (Figure 3)

Using the flood map is worth 0.27 over ignoring it. Lookahead scores 0.550 on
test; PPO scores 0.521, a paired difference of -0.029 [-0.039, -0.019], and
-0.024 [-0.041, -0.007] on Bolivia. PPO is level with greedy and assignment.
Executing PPO's most likely move collapses the score to 0.290 because teams
share one actor and start with identical views.

### 5.6 Chain ablation (Figure 4)

With the planner fixed, replacing no map by the radar ensemble gains 0.268.
Every further refinement is worth about 0.01 or less: thresholded map -0.009,
single model -0.004, calibrated 0.000, uncertainty weight -0.001. Optical or
fused input gains about 0.01 in all three settings, and the truth 0.017-0.023.
Calibration and uncertainty reach intervals that exclude zero only in the
hardest setting, at +0.002 each. The tuned uncertainty weight changes sign
across settings.

### 5.7 Dispatch with costly mistakes (Figure 5)

The task now separates methods: the gap from the radar planner to the truth is
0.05-0.09 on test and 0.09-0.23 on Bolivia. A calibrated chance of a dry cell
predicts dry cells better than the map at face value (Brier 0.084 against
0.152 on test, 0.085 against 0.113 on Bolivia), and on test it is worth more
as the cost rises: +0.013 [-0.004, +0.033], +0.039 [0.000, +0.086] and +0.090
[+0.017, +0.184] at costs 0.25, 0.5 and 1. Wasted deployments fall from 1.7 to
0.4-0.9 per episode. On Bolivia the face-value planner scores higher at every
cost, though no difference is significant on 11 scenarios. Adding ensemble
disagreement to the calibration helps once, +0.021 [+0.002, +0.044] at cost 1
on test, and in none of the other five cases. The optical model, calibrated
the same way, is the largest gain on Bolivia (+0.06 to +0.19).

PPO scores 0.404 on test and 0.327 on Bolivia, against 0.596 and 0.511 for
the calibrated planner: differences of -0.192 [-0.254, -0.132] and -0.184
[-0.258, -0.105]. It deploys less and wastes more. Its training score was flat
from half-way, and it loses 0.06 from its training scenarios to test. Without
the uncertainty input it does no worse (0.424 and 0.337). With a team
identifier, deterministic execution no longer collapses (0.37 against 0.40 for
sampled moves).

### 5.8 Question answering (RQ5)

| System | Accuracy | Answerable correct | Unanswerable refused | Decimals not in cited records |
|---|---|---|---|---|
| TerraCore | 0.78 | 0.89 | 0.42 | 0.007 |
| TerraCore, no critic | 0.78 | 0.86 | 0.50 | 0.007 |
| Fixed pipeline | 0.43 | 0.57 | 0.00 | 0.000 |
| LLM agent (llama3:8b) | 0.67 | 0.57 | 1.00 | 0.093 |

TerraCore beats the fixed pipeline by 0.35 [0.20, 0.49] but is not
distinguishable from the LLM agent (+0.10 [-0.10, +0.31]). The two fail in
opposite directions: TerraCore answers most answerable questions but refuses
only 5 of 12 unanswerable ones, returning true but irrelevant benchmark
figures for questions about deaths, rainfall or places outside the corpus;
the LLM refuses all 12 but also refuses 5 answerable questions, picks the
wrong one of two similar records, and cannot count scenes from 8 records.
Removing the critic changes accuracy by 0.00 [-0.06, +0.06].

These failures were found on that set, so it became the development set. A
second set of 49 questions with new wording, targets and 16 new unanswerable
questions was frozen before TerraCore was changed. Fixes developed on the
first set only (refuse names absent from the store, no default answer for an
unrecognised question type, look up named policies and datasets) raise
held-out accuracy from 0.59 to 0.88 (+0.29 [0.16, 0.41]) and refusals of
unanswerable questions from 2 to 13 of 16. On the held-out set TerraCore scores
0.88 and the LLM agent 0.78 (+0.10 [-0.04, +0.24]); the critic again changes
nothing. Calibration questions all fail: no agent looks them up for a named
place.

### 5.9 Further experiments

**Foundation model (E2).** Prithvi-EO-2.0-100M-TL, fine-tuned with a small
decoder on six optical bands, reaches 0.817 test and 0.691 Bolivia IoU (3
seeds), against 0.839 and 0.782 for a U-Net on the same bands; frozen, 0.801
and 0.662. The pretrained model transfers worse to the unseen region.

**Missing sensors (E1).** The early-fusion model scores 0.280 on test and 0.029
on Bolivia with the optical bands removed, far below a radar-only model (0.657,
0.640). Modality dropout restores the optical-only case (0.823) but lifts
radar-only only to 0.508. Late fusion with a weight tuned on validation puts
0.9 on optical and matches optical alone.

**Learned dispatch policies (RQ6).** Imitating the calibrated planner then
fine-tuning with PPO reaches 0.575 utility on test against the planner's 0.596
(-0.021 [-0.047, +0.005]); imitation alone 0.455, PPO from scratch 0.404, a
MAPPO global critic 0.559.

**Forecasting (E3).** For GloFAS discharge at the 15 event sites, persistence
has median NSE 0.985, 0.941 and 0.823 at 1, 3 and 7 days; a GRU on 30 days of
discharge and rain lowers squared error by 27%, 17% and 16%. GloFAS itself
exceeds its flood threshold at only 10 of 13 test-period events.

**Flooded roads (E10).** On 813k OSM road points, thresholding the
out-of-event radar flood probability gives F1 0.626 on test; a per-point MLP
with terrain and road features 0.617, GraphSAGE over the road graph 0.599.

## 6. Discussion

The chain works mechanically, but its premise is borne out only in part. In
the first task perception error was nearly free: need is smooth and averaged
over cells, and a wrong visit cost a step. Making mistakes costly (Section
5.7) confirmed that this was why calibration did not matter: once a dry
deployment costs something, a calibrated chance of a dry cell is worth up to
0.09 on the test events.

Two findings limit the claim. First, the gain does not transfer: a calibration
fitted on validation events is no better than none on Bolivia, mirroring the
pixel-level result that temperature scaling half-fixes calibration there. A
decision layer that relies on calibrated probabilities inherits the
calibration's dependence on region. Second, ensemble disagreement adds almost
nothing to a calibration of the mean prediction. Its signal is small and
concentrated at water edges, and a fitted model of the mean already captures
most of what predicts a dry cell.

The evidence layer shows a parallel gap. TerraCore's critic guarantees that
every number it reports is in a cited record, and it does: 0.7% of its
decimals are not in a cited record, all of them means it computed itself. But
faithfulness is not relevance. The critic never asks whether a claim answers
the question, so TerraCore answers questions it should refuse with true,
cited, irrelevant numbers. A small LLM given the same records is the opposite:
cautious to a fault, but loose with which record it uses. A system that
deserves trust needs both checks, and our results do not show that splitting
the work into specialised agents is what delivers either.

The learned policy is weaker than planning in both tasks. From scratch, with
45 training scenarios, it partly memorises them; started from the planner by
imitation, RL recovers most of the gap but finds nothing the planner misses.

The other additions repeat one pattern: more machinery on top of a decent flood
map rarely helps. A large pretrained encoder transfers worse than a small
network trained on the task; a road graph adds nothing to the per-point flood
probability; a learned forecaster barely beats persistence. Where something
does help, it is better input (optical over radar), or a system design that
keeps a dedicated radar model for when optical data are missing, rather than
one fused model.

What a fuller test still needs: need weighted by exposure, travel costs
through flooded cells, forecasts rather than snapshots, recalibration on the
unseen region from a few labelled cells, and an unseen region with more than
one event.

## 7. Limitations

- One benchmark, 446 chips; the unseen region is a single event of 15 chips,
  so no event-level interval exists for it.
- One foundation model (100M, one recipe, no time/location inputs); forecasts
  target simulated GloFAS discharge, not gauges; road labels are pixel water at
  the road point, not passability.
- Labels include permanent water and exclude cloudy pixels, which removes the
  case radar exists for.
- Leave-one-event-out uses one seed per fold and more training chips than the
  standard models.
- The retrieval benchmark is generated from the corpus; weights and provenance
  scores are hand-set.
- The simulation is invented: no population, roads or elevation. In the
  dispatch task the waste cost and the 5% dry threshold are arbitrary, and
  the single uncertainty gain would not survive a correction for six
  comparisons.
- RL tuning used one seed on 10 held-out scenarios; the gain seen there did
  not hold on the test set.
- The RQ5 questions were written by the author of TerraCore's rules and their
  gold facts come from the store the systems read; 49 questions and one small
  local LLM. The held-out set was written by the same person after seeing
  the first set's failures.
- Dataset licences other than Sen1Floods11's absence of one are unverified.

## 8. Reproducibility

`README.md` lists every command. `terramind validate` regenerates
`docs/results.md`, `experiments/results.json` and the figures from run files
and re-evaluates the verdicts in Table 1 as rules over those numbers. Hardware:
one 8 GB laptop GPU. The training runs reported here were made before the
first commit, so their reports record no commit hash.

## Figures

1. `figures/fig1-per-event-iou.png`: IoU by event and input.
2. `figures/fig2-leave-one-event-out.png`: seen against held-out IoU.
3. `figures/fig3-policies-simulated.png`: simulated policy comparison.
4. `figures/fig4-chain-ablation-simulated.png`: simulated chain ablation.
5. `figures/fig5-dispatch-simulated.png`: simulated dispatch, utility as the
   waste cost rises.
