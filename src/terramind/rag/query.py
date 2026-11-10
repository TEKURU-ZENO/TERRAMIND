"""Rule-based query parsing: places, time window and intent.

Places come from a gazetteer built from the evidence corpus itself (flood
events), so the parser can only resolve regions the system has data for. A
question about anywhere else resolves to no region, and the agents then say so
instead of guessing coordinates.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date

from terramind.evidence.store import Evidence

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

ALIASES = {
    "event:s1f11-Sri-Lanka": ["sri lanka", "sri-lanka"],
    "event:s1f11-Mekong": ["mekong", "cambodia"],
    "event:s1f11-USA": ["usa", "united states"],
    "event:chennai-2015": ["chennai"],
    "event:kerala-2018": ["kerala"],
    "event:ahr-2021": ["ahr", "germany"],
    "event:sindh-2022": ["sindh"],
}

INTENT_KEYWORDS = {
    "scene": ["scene", "imagery", "image", "satellite", "sentinel", "acquisition", "pass", "cloud"],
    "benchmark": [
        "iou", "accura", "perform", "how well", "model", "score", "calibrat", "uncertain",
        "confiden", "reliab", "transfer", "generali", "fusion", "predict",
    ],
    "dataset": ["licen", "license", "dataset", "redistribut", "attribution", "source data"],
    "chip": ["chip", "label", "water fraction", "how much water", "flooded"],
    "simulation": [
        "simulat", "policy", "policies", "teams", "allocat", "planner", "ppo", "reinforcement",
        "response strateg", "what if", "what happens if", "decision",
    ],
}  # fmt: skip


@dataclass
class Region:
    name: str
    evidence_id: str
    bbox: list[float]


@dataclass
class ParsedQuery:
    text: str
    regions: list[Region] = field(default_factory=list)
    start: str | None = None
    end: str | None = None
    kinds: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)

    @property
    def bboxes(self) -> list[list[float]]:
        return [r.bbox for r in self.regions]


def build_gazetteer(corpus: list[Evidence]) -> dict[str, Region]:
    """Lower-case place name -> region, from event evidence."""
    gazetteer: dict[str, list[Region]] = {}
    for e in corpus:
        if e.kind != "event" or not e.bbox:
            continue
        names = list(ALIASES.get(e.id, []))
        if e.id.startswith("event:s1f11-"):
            names.append(e.id.removeprefix("event:s1f11-").lower())
        for name in set(names):
            gazetteer.setdefault(name, []).append(Region(name, e.id, e.bbox))
    # A name can refer to more than one event; keep them all under suffixed keys.
    flat = {}
    for name, regions in gazetteer.items():
        for i, region in enumerate(regions):
            flat[name if i == 0 else f"{name}#{i}"] = region
    return flat


def _time_window(text: str) -> tuple[str | None, str | None]:
    iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso:
        days = sorted(date(int(y), int(m), int(d)) for y, m, d in iso)
        return days[0].isoformat(), days[-1].isoformat()
    month = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b", text)
    if month and month.group(1).lower() in MONTHS:
        y, m = int(month.group(2)), MONTHS[month.group(1).lower()]
        return date(y, m, 1).isoformat(), date(y, m, calendar.monthrange(y, m)[1]).isoformat()
    year = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year:
        return f"{year.group(1)}-01-01", f"{year.group(1)}-12-31"
    return None, None


def parse(text: str, gazetteer: dict[str, Region]) -> ParsedQuery:
    lower = text.lower()
    regions, seen = [], set()
    for key, region in gazetteer.items():
        name = key.split("#")[0]
        if (
            re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", lower)
            and region.evidence_id not in seen
        ):
            regions.append(region)
            seen.add(region.evidence_id)
    start, end = _time_window(text)
    kinds = [kind for kind, words in INTENT_KEYWORDS.items() if any(w in lower for w in words)]
    modalities = []
    if re.search(r"\b(radar|sar|sentinel-1|s1)\b", lower):
        modalities.append("s1")
    if re.search(r"\b(optical|sentinel-2|s2)\b", lower):
        modalities.append("s2")
    if "fusion" in lower or "fused" in lower:
        modalities.append("s1s2")
    return ParsedQuery(text, regions, start, end, kinds, modalities)
