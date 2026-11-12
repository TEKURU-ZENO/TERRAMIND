# Reading list and novelty audit

The planning document cites these as prior work. By 2026-11-12 all eight were
confirmed to exist with the reference below, from search results, abstracts and
listing pages only. **None has been read in full**, so the "Read" box stays
empty and no claim about their content beyond the abstract should be made.

| Read | Work | Claimed in plan | Verified reference |
|---|---|---|---|
| [ ] | Prithvi-EO-2.0 | Multitemporal EO foundation model, 4.2M samples | Exists: arXiv:2412.02732 (Szwarcman et al., Dec 2024). The 4.2M-sample figure matches the arXiv listing. |
| [ ] | PANGAEA | Global geospatial foundation model benchmark | Exists: arXiv:2412.04204 (Marsocci et al., Dec 2024). Its listing reports that foundation models do not consistently beat a supervised U-Net, which bears on our U-Net baseline. |
| [ ] | GEO-Bench | Foundation models for Earth monitoring (NeurIPS) | Exists: arXiv:2306.03831 (Lacoste et al., 2023). NeurIPS venue not confirmed. |
| [ ] | Sen1Floods11 | Bonafilia et al., CVPR Workshops 2020 | Exists: Bonafilia, Tellman, Anderson, Issenberg, CVPR Workshops 2020, pp. 210-211. The GitHub README states no licence. How the hand labels were drawn could not be checked (paper PDF not retrievable). |
| [ ] | Spatial-RAG | "Findings of ACL 2026" (check venue and year) | Exists: arXiv:2502.18470 (Yu et al., Feb 2025); listed in the ACL Anthology as 2026.findings-acl.539. A hybrid spatial + semantic retriever with multi-objective ranking, so our hybrid score is not novel. |
| [ ] | GeoGraphRAG | 2025, ScienceDirect (check) | Exists: Liang, Hou, Jiao, Qing, Zhao, Shen, Xiang, Wu (2025), "GeoGraphRAG: a graph-based retrieval-augmented generation approach for empowering large language models in automated geospatial modeling", Int. J. Applied Earth Observation and Geoinformation 142, 104712. Graph RAG for geospatial *modelling and code generation*, not evidence QA. |
| [ ] | GeoAgentic-RAG | 2026, ScienceDirect (check) | Exists: Liang et al. (2026), "GeoAgentic-RAG: a multi-agent framework for autonomous geospatial reasoning and visual insight generation with LLM" (indexed in FAO AGRIS; venue not confirmed). Multi-agent semantic-spatial retrieval plus executable analysis, Nanjing and Guangzhou; reports 88.3% answer correctness. Closest prior work to TerraCore; weakens any claim that a multi-agent geospatial RAG is new. |
| [ ] | Hanny et al. | 2026, multimodal disaster detection (check) | Exists: Hanny et al. (2026), Natural Hazards: disaster detection from Bluesky posts, GDELT headlines and weather data with a generative language model on an H3 grid, validated against satellite-derived references (2024 Central Europe floods, 2025 Southern California wildfires). Related AGILE GIScience Series 2026 paper by Hanny et al. on relevance scoring. Different input (social/news media) from ours. |

## For each paper, record

- What it actually does, and on which data and regions.
- Which TERRA MIND claim it weakens or rules out.
- Whether code and weights are available as a baseline.

## Novelty hypotheses to test against the literature

| Hypothesis | Prior confidence |
|---|---|
| Cross-region multimodal fusion with calibrated uncertainty | Medium |
| Spatial-temporal evidence provenance for decisions | Medium |
| Uncertainty-conditioned RL policy | Medium |
| Integrated prediction-to-action benchmark | Medium-high |
| Generic multi-agent GIS assistant | Low; do not claim |
