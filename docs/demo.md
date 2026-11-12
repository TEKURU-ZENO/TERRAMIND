# Demo walkthrough

About ten minutes. It needs the data and runs already on this machine
(`data/`, `experiments/**/runs/`); nothing is downloaded or trained.

## Start

```bash
.venv/Scripts/terramind serve
```

```bash
npm --prefix apps/web run dev
```

Open http://localhost:3000.

## 1. Perception and where it fails (Benchmark tab)

- Click **Pakistan** in the event list, then a test chip. Radar IoU is near 0.
  Switch imagery to Optical and tick **Hand label**: the water is visible to the
  optical sensor but not in the radar image.
- Click **Nigeria** and a chip: the same model scores above 0.8.
- Tick **Ensemble disagreement** to see where the five models differ.

Point: one pooled number (0.65) hides events from 0.18 to 0.86.

## 2. One chip through the whole chain (chip card)

- With a Pakistan test chip selected, press **Build report**. Reliability is
  low and no allocation is recommended, with the benchmark records it rests on.
- Do the same for a Bolivia chip: reliability good, with a simulated response.

## 3. Simulated response (Simulate tab)

- With a Bolivia chip selected, run **Ignore the model**, then **Lookahead**,
  then **PPO (learned)**. Compare "Average over the episode".
- Switch the blue cells between labelled water and the model prediction.

Point: using the map matters a lot; the learned policy does not beat planning.
Say that this is simulated.

## 4. Evidence-grounded answers (Ask tab)

- "How well does the flood model perform in Bolivia?" Answer with citations.
- "Which response policy works best in the simulation for Bolivia?" Answer
  marked simulated.
- "Which roads in Mumbai will flood in the next 48 hours?" It abstains.

## 5. Numbers

```bash
.venv/Scripts/terramind validate
```

Prints the verdict for each research question and the protocol checks, and
rewrites `docs/results.md` and the figures in `paper/figures/`.

## What not to claim

- Nothing here is a field result; the response problem is invented.
- No forecasting, no population or road data, no foundation-model baseline.
- Sen1Floods11 states no licence in its repository: do not hand out the data.
