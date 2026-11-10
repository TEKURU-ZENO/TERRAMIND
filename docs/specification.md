# TERRA MIND specification (condensed)

Condensed from the planning document "Terra Mind Geospatial Intelligence
Ecosystem". This file is the working reference; update it when decisions change.

## Decisions

| Dimension | Decision |
|---|---|
| Domain | Hybrid: disaster + climate + urban. **First track: global flood intelligence** |
| Ambition | Research-grade engineering product |
| Geography | Global, validated on selected regions with labels |
| Architecture | Modular monolith first, split services later |
| Hardware | RTX 5050 laptop, 8 GB VRAM; no foundation-model pretraining |

## Central research contribution

Uncertainty-aware, evidence-grounded geospatial decision intelligence: how
uncertainty and evidence should propagate from perception through simulation to
action, and whether that coupling measurably improves decision quality under
geographic distribution shift.

Everything else supports this. A generic multi-agent GIS assistant is **not** a
novelty claim.

## Subsystems

| Level | System | Sprint 0 |
|---|---|---|
| Observation | Earth data infrastructure | started |
| Understanding | GeoFusion (multimodal encoder) | SAR-only baseline |
| Relationships | TerraGraph | no |
| Evidence | TerraRAG | no |
| Coordination | TerraCore (LangGraph agents) | no |
| Experimentation | TerraSim | no |
| Decision-making | TerraPolicy (PPO / MAPPO) | no |
| Trust | TerraTrust (uncertainty, provenance) | provenance in manifests and reports |
| Experience | Dashboard | no |

## Research questions

1. RQ1 Does optical + SAR + weather + terrain fusion beat single modalities?
2. RQ2 Does it generalise to unseen regions, climates and sensors?
3. RQ3 Can uncertainty identify when the system should abstain?
4. RQ4 Does semantic + spatial + temporal + graph retrieval improve geospatial QA?
5. RQ5 Are specialised agents more reliable than one agent or a fixed workflow?
6. RQ6 Does MARL beat heuristic and optimisation baselines in simulation?
7. RQ7 Does the integrated chain improve decision quality over separate parts?

## Experiments

E1 fusion, E2 foundation-model comparison, E3 forecasting, E4 geographic
transfer, E5 spatial-temporal RAG, E6 agentic workflows, E7 RL, E8 end-to-end,
plus ablations A0-A10.

## Evaluation rules

- Geographic holdouts; no random splits of adjacent patches.
- 3-5 seeds; bootstrap confidence intervals at event or region level.
- Report per-region and worst-region performance, and negative results.
- Simulated outcomes are labelled as simulated.

## Milestones (24 weeks)

| Weeks | Milestone | Deliverable |
|---|---|---|
| 1-3 | M1 Research and data foundation | Reproducible dataset pipeline |
| 4-6 | M2 Geospatial platform | Earth Explorer |
| 7-10 | M3 Multimodal ML | Fusion, flood segmentation, uncertainty |
| 11-13 | M4 Spatiotemporal intelligence | Forecasting, GNN, transfer |
| 14-16 | M5 Spatial RAG and agents | Reasoning system |
| 17-19 | M6 Simulation and RL | TerraSim, PPO/MAPPO |
| 20-22 | M7 Integration | Alpha |
| 23-24 | M8 Validation | Paper package and demo |
