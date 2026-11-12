"use client";

import maplibregl, { type GeoJSONSource, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useMemo, useRef, useState } from "react";

type Bounds = [number, number, number, number];

type ChipProps = {
  id: string;
  event: string;
  split: string;
  iou: number | null;
  precision: number | null;
  recall: number | null;
  water_fraction: number;
  labelled_fraction: number;
  mean_uncertainty: number | null;
};

type ChipFeature = {
  type: "Feature";
  id: string;
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: ChipProps;
};

type ChipCollection = {
  type: "FeatureCollection";
  model: { name: string; modality: string; seeds: number[] };
  note: string;
  features: ChipFeature[];
};

type SummaryRow = {
  event: string;
  split: string;
  chips: number;
  mean_chip_iou: number | null;
  bounds: Bounds;
};

type FloodEvent = {
  id: string;
  name: string;
  region: string;
  bbox: Bounds;
  start: string;
  end: string;
  confirmed: boolean;
};

type Scene = {
  id: string;
  collection: string;
  catalog: string;
  datetime: string | null;
  bbox: Bounds | null;
  cloud_cover: number | null;
  platform: string | null;
};

type AskReport = {
  question: string;
  verdict: "answer" | "abstain";
  answer: string;
  caveats: string[];
  evidence: {
    n: number;
    id: string;
    kind: string;
    title: string;
    provenance: number;
    bbox: Bounds | null;
    start: string | null;
  }[];
  rejected_claims: { claim: string; reason: string }[];
  regions: { name: string; bbox: Bounds }[];
  trace: { agent: string; action: string; detail: unknown }[];
};

type Situation = {
  chip: string;
  reliability: string;
  abstain: boolean;
  recommendation: string;
  flags: string[];
  sections: { title: string; statements: { text: string; evidence: string[]; simulated: boolean }[] }[];
  caveats: string[];
  summary?: { text: string | null; accepted: boolean; reason: string | null };
};

type SimEpisode = {
  chip: string;
  grid: number;
  teams: number;
  horizon: number;
  truth: number[][];
  prediction: number[][];
  positions: [number, number][][];
  served: number[];
  auc: number;
};

const POLICIES: [string, string][] = [
  ["blind-lookahead", "Ignore the model"],
  ["greedy", "Greedy"],
  ["assignment", "Assignment"],
  ["lookahead", "Lookahead"],
  ["ppo", "PPO (learned)"],
  ["oracle-lookahead", "Oracle (sees truth)"],
];
const TEAM_COLOURS = ["#f472b6", "#fbbf24", "#a3e635"];

const EXAMPLES = [
  "How well does the flood model perform in Bolivia?",
  "Which Sentinel-2 scenes cover the Kerala floods in August 2018?",
  "Is the optical model's confidence reliable on unseen regions?",
  "Which response policy works best in the simulation for Bolivia?",
  "What is the licence for OpenStreetMap data?",
  "Which roads in Mumbai will flood in the next 48 hours?",
];

const SPLITS = ["all", "train", "valid", "test", "bolivia"] as const;
const BASES = ["none", "sar", "optical"] as const;
const OVERLAYS = ["label", "prediction", "errors", "uncertainty"] as const;
type Base = (typeof BASES)[number];
type Overlay = (typeof OVERLAYS)[number];

const OVERLAY_INFO: Record<Overlay, { label: string; swatches: [string, string][] }> = {
  label: { label: "Hand label", swatches: [["#38bdf8", "water"], ["#94a3b8", "no data"]] },
  prediction: { label: "Model prediction", swatches: [["#facc15", "predicted water"]] },
  errors: {
    label: "Errors",
    swatches: [["#ef4444", "false alarm"], ["#3b82f6", "missed water"]],
  },
  uncertainty: { label: "Ensemble disagreement", swatches: [["#d946ef", "members disagree"]] },
};

// Sequential ramp for IoU (0 = dark, 1 = bright), readable on a dark basemap.
const IOU_COLOUR = [
  "case",
  ["==", ["get", "iou"], null],
  "#64748b",
  ["interpolate", ["linear"], ["get", "iou"], 0, "#5b21b6", 0.5, "#0d9488", 1, "#fde047"],
] as unknown as maplibregl.ExpressionSpecification;

const fmt = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined ? "n/a" : v.toFixed(digits);
const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(1)}%`;

function boundsOf(f: ChipFeature): Bounds {
  const ring = f.geometry.coordinates[0];
  return [ring[0][0], ring[0][1], ring[2][0], ring[2][1]];
}

function bboxPolygon(b: Bounds) {
  const [w, s, e, n] = b;
  return {
    type: "Polygon" as const,
    coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
  };
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json();
}

export default function Explorer() {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [tab, setTab] = useState<"benchmark" | "events" | "ask" | "simulate">("benchmark");
  const [view3d, setView3d] = useState(false);
  const [policy, setPolicy] = useState("lookahead");
  const [episode, setEpisode] = useState<SimEpisode | null>(null);
  const [simStep, setSimStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [situation, setSituation] = useState<Situation | null>(null);
  const [loadingSituation, setLoadingSituation] = useState(false);
  const [simShow, setSimShow] = useState<"truth" | "prediction">("truth");
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [report, setReport] = useState<AskReport | null>(null);
  const [chips, setChips] = useState<ChipCollection | null>(null);
  const [summary, setSummary] = useState<SummaryRow[]>([]);
  const [events, setEvents] = useState<FloodEvent[]>([]);

  const [split, setSplit] = useState<(typeof SPLITS)[number]>("all");
  const [eventFilter, setEventFilter] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [base, setBase] = useState<Base>("sar");
  const [overlays, setOverlays] = useState<Overlay[]>(["errors"]);
  const [opacity, setOpacity] = useState(0.85);

  const [activeEvent, setActiveEvent] = useState<string | null>(null);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneIndex, setSceneIndex] = useState(0);

  // Load data once.
  useEffect(() => {
    Promise.all([
      getJson<ChipCollection>("/api/chips"),
      getJson<{ rows: SummaryRow[] }>("/api/summary"),
      getJson<FloodEvent[]>("/api/events"),
    ])
      .then(([c, s, e]) => {
        setChips(c);
        setSummary(s.rows);
        setEvents(e);
      })
      .catch((err) => setError(`Could not reach the API. Is "terramind serve" running? (${err.message})`));
  }, []);

  // Create the map once.
  useEffect(() => {
    if (!container.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: container.current,
      center: [20, 15],
      zoom: 1.6,
      attributionControl: { compact: true },
      style: {
        version: 8,
        sources: {
          basemap: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            maxzoom: 19,
            attribution: "© OpenStreetMap contributors",
          },
          // Public terrain tiles on AWS Open Data (Terrarium encoding), for the 3D view.
          dem: {
            type: "raster-dem",
            tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
            tileSize: 256,
            maxzoom: 14,
            encoding: "terrarium",
            attribution: "Terrain: Mapzen Terrain Tiles on AWS (SRTM, GMTED, ETOPO1 and others)",
          },
        },
        layers: [
          { id: "background", type: "background", paint: { "background-color": "#0b1120" } },
          {
            id: "basemap",
            type: "raster",
            source: "basemap",
            // Dimmed and desaturated so data layers stand out on the dark UI.
            paint: {
              "raster-saturation": -0.85,
              "raster-brightness-max": 0.5,
              "raster-contrast": 0.15,
            },
          },
          {
            id: "hillshade",
            type: "hillshade",
            source: "dem",
            layout: { visibility: "none" },
            paint: { "hillshade-exaggeration": 0.35, "hillshade-shadow-color": "#020617" },
          },
        ],
      },
    });
    map.on("error", (e) => console.error("map error:", e.error?.message ?? e));
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({}), "bottom-right");
    map.on("load", () => {
      const empty = { type: "FeatureCollection" as const, features: [] };
      map.addSource("chips", { type: "geojson", data: empty, promoteId: "id" });
      map.addSource("chip-points", { type: "geojson", data: empty });
      map.addSource("scenes", { type: "geojson", data: empty });
      map.addSource("event-boxes", { type: "geojson", data: empty });
      map.addSource("sim-cells", { type: "geojson", data: empty });
      map.addSource("sim-trails", { type: "geojson", data: empty });
      map.addSource("sim-teams", { type: "geojson", data: empty });

      map.addLayer({
        id: "event-boxes",
        type: "line",
        source: "event-boxes",
        paint: { "line-color": "#f8fafc", "line-width": 1.5, "line-dasharray": [3, 2] },
      });
      map.addLayer({
        id: "scenes-fill",
        type: "fill",
        source: "scenes",
        paint: {
          "fill-color": "#38bdf8",
          "fill-opacity": ["case", ["get", "active"], 0.2, 0],
        },
      });
      map.addLayer({
        id: "scenes-line",
        type: "line",
        source: "scenes",
        paint: {
          "line-color": "#38bdf8",
          "line-width": ["case", ["get", "active"], 2, 0.5],
          "line-opacity": ["case", ["get", "active"], 1, 0.35],
        },
      });
      map.addLayer({
        id: "chip-points",
        type: "circle",
        source: "chip-points",
        maxzoom: 9,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 3, 8, 6],
          "circle-color": IOU_COLOUR,
          "circle-stroke-color": "#0b1120",
          "circle-stroke-width": 1,
        },
      });
      map.addLayer({
        id: "chips-fill",
        type: "fill",
        source: "chips",
        minzoom: 8,
        paint: {
          "fill-color": IOU_COLOUR,
          "fill-opacity": ["case", ["boolean", ["feature-state", "selected"], false], 0, 0.45],
        },
      });
      map.addLayer({
        id: "chips-line",
        type: "line",
        source: "chips",
        minzoom: 8,
        paint: {
          "line-color": ["case", ["boolean", ["feature-state", "selected"], false], "#f8fafc", "#0b1120"],
          "line-width": ["case", ["boolean", ["feature-state", "selected"], false], 2, 0.75],
        },
      });

      map.addLayer({
        id: "sim-cells",
        type: "fill",
        source: "sim-cells",
        paint: {
          "fill-color": ["case", ["get", "served"], "#34d399", "#38bdf8"],
          "fill-opacity": ["get", "opacity"],
        },
      });
      map.addLayer({
        id: "sim-trails",
        type: "line",
        source: "sim-trails",
        paint: { "line-color": ["get", "colour"], "line-width": 2, "line-opacity": 0.8 },
      });
      map.addLayer({
        id: "sim-teams",
        type: "circle",
        source: "sim-teams",
        paint: {
          "circle-radius": 6,
          "circle-color": ["get", "colour"],
          "circle-stroke-color": "#0b1120",
          "circle-stroke-width": 2,
        },
      });

      const pick = (e: maplibregl.MapLayerMouseEvent) => {
        const id = e.features?.[0]?.properties?.id;
        if (id) setSelectedId(id as string);
      };
      for (const layer of ["chips-fill", "chip-points"]) {
        map.on("click", layer, pick);
        map.on("mouseenter", layer, () => (map.getCanvas().style.cursor = "pointer"));
        map.on("mouseleave", layer, () => (map.getCanvas().style.cursor = ""));
      }
      setReady(true);
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  const visible = useMemo(() => {
    if (!chips) return [];
    return chips.features.filter(
      (f) =>
        (split === "all" || f.properties.split === split) &&
        (!eventFilter || f.properties.event === eventFilter),
    );
  }, [chips, split, eventFilter]);

  const selected = useMemo(
    () => chips?.features.find((f) => f.id === selectedId) ?? null,
    [chips, selectedId],
  );

  // Push filtered chips to the map.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const show = tab === "benchmark" || tab === "simulate" ? visible : [];
    (map.getSource("chips") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: show,
    } as GeoJSON.FeatureCollection);
    (map.getSource("chip-points") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: show.map((f) => {
        const [w, s, e, n] = boundsOf(f);
        return {
          type: "Feature",
          geometry: { type: "Point", coordinates: [(w + e) / 2, (s + n) / 2] },
          properties: f.properties,
        };
      }),
    } as GeoJSON.FeatureCollection);
  }, [ready, visible, tab]);

  // Selected chip: outline state and image overlays.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const wanted: string[] = [];
    if (selected && (tab === "benchmark" || tab === "simulate")) {
      map.setFeatureState({ source: "chips", id: selected.id }, { selected: true });
      const [w, s, e, n] = boundsOf(selected);
      const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [
        [w, n],
        [e, n],
        [e, s],
        [w, s],
      ];
      // The simulation draws its own cells, so only the imagery goes underneath it.
      const stack = [
        ...(base === "none" ? [] : [base]),
        ...(tab === "simulate" ? [] : OVERLAYS.filter((o) => overlays.includes(o))),
      ];
      for (const layer of stack) {
        const key = `overlay-${layer}`;
        wanted.push(key);
        map.addSource(key, {
          type: "image",
          url: `${window.location.origin}/api/chips/${selected.id}/${layer}.png`,
          coordinates,
        });
        map.addLayer(
          {
            id: key,
            type: "raster",
            source: key,
            paint: {
              "raster-opacity": layer === base ? 1 : opacity,
            },
          },
          "chips-line",
        );
      }
    }
    return () => {
      // On unmount the map has already been removed; there is nothing left to clean.
      if (mapRef.current !== map) return;
      for (const key of wanted) {
        if (map.getLayer(key)) map.removeLayer(key);
        if (map.getSource(key)) map.removeSource(key);
      }
      if (selected && map.getSource("chips")) {
        map.setFeatureState({ source: "chips", id: selected.id }, { selected: false });
      }
    };
  }, [ready, selected, base, overlays, opacity, tab, visible]);

  // Candidate event boxes and scene footprints.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const onEvents = tab === "events";
    (map.getSource("event-boxes") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: onEvents
        ? events.map((e) => ({ type: "Feature", geometry: bboxPolygon(e.bbox), properties: {} }))
        : [],
    } as GeoJSON.FeatureCollection);
    (map.getSource("scenes") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: onEvents
        ? scenes
            .map((s, i) => ({ s, i }))
            .filter(({ s }) => s.bbox)
            // Draw the active scene last so it sits on top.
            .sort((a, b) => Number(a.i === sceneIndex) - Number(b.i === sceneIndex))
            .map(({ s, i }) => ({
              type: "Feature",
              geometry: bboxPolygon(s.bbox as Bounds),
              properties: { active: i === sceneIndex },
            }))
        : [],
    } as GeoJSON.FeatureCollection);
  }, [ready, tab, events, scenes, sceneIndex]);

  // Simulation playback: cells, trails and teams up to the current step.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const cells: GeoJSON.Feature[] = [];
    const trails: GeoJSON.Feature[] = [];
    const teams: GeoJSON.Feature[] = [];
    if (tab === "simulate" && episode && selected && selected.id === episode.chip) {
      const [w, s, e, n] = boundsOf(selected);
      const g = episode.grid;
      const dx = (e - w) / g;
      const dy = (n - s) / g;
      const visited = new Set<number>();
      for (const step of episode.positions.slice(0, simStep + 1))
        for (const [r, c] of step) visited.add(r * g + c);
      const values = simShow === "truth" ? episode.truth : episode.prediction;
      for (let r = 0; r < g; r++)
        for (let c = 0; c < g; c++) {
          const served = visited.has(r * g + c);
          const opacity = Math.min(0.85, values[r][c] * 0.8 + (served ? 0.12 : 0));
          if (opacity < 0.02) continue;
          cells.push({
            type: "Feature",
            geometry: bboxPolygon([w + c * dx, n - (r + 1) * dy, w + (c + 1) * dx, n - r * dy]),
            properties: { served, opacity },
          });
        }
      for (let k = 0; k < episode.teams; k++) {
        // Teams sharing a cell are nudged apart so all stay visible.
        const off = (k - (episode.teams - 1) / 2) * 0.22;
        const at = ([r, c]: [number, number]) => [w + (c + 0.5 + off) * dx, n - (r + 0.5 + off) * dy];
        const path = episode.positions.slice(0, simStep + 1).map((step) => at(step[k]));
        const colour = TEAM_COLOURS[k % TEAM_COLOURS.length];
        if (path.length > 1)
          trails.push({
            type: "Feature",
            geometry: { type: "LineString", coordinates: path },
            properties: { colour },
          });
        teams.push({
          type: "Feature",
          geometry: { type: "Point", coordinates: path[path.length - 1] },
          properties: { colour },
        });
      }
    }
    const set = (id: string, features: GeoJSON.Feature[]) =>
      (map.getSource(id) as GeoJSONSource).setData({ type: "FeatureCollection", features });
    set("sim-cells", cells);
    set("sim-trails", trails);
    set("sim-teams", teams);
  }, [ready, tab, episode, simStep, simShow, selected]);

  useEffect(() => {
    if (!playing || !episode) return;
    const timer = setInterval(() => {
      setSimStep((t) => {
        if (t >= episode.horizon) {
          setPlaying(false);
          return t;
        }
        return t + 1;
      });
    }, 180);
    return () => clearInterval(timer);
  }, [playing, episode]);

  const simulate = (chipId: string, name: string) => {
    setPolicy(name);
    setError(null);
    setPlaying(false);
    getJson<SimEpisode>(`/api/sim/episode?chip=${encodeURIComponent(chipId)}&policy=${name}`)
      .then((r) => {
        setEpisode(r);
        setSimStep(0);
        setPlaying(true);
      })
      .catch((err) => setError(err.message));
  };

  const loadSituation = (chipId: string, narrate = false) => {
    setLoadingSituation(true);
    setError(null);
    getJson<Situation>(`/api/report/${encodeURIComponent(chipId)}${narrate ? "?narrate=true" : ""}`)
      .then(setSituation)
      .catch((err) => setError(err.message))
      .finally(() => setLoadingSituation(false));
  };

  // 3D: globe projection, terrain and hillshade.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    map.setProjection({ type: view3d ? "globe" : "mercator" });
    map.setTerrain(view3d ? { source: "dem", exaggeration: 1.5 } : null);
    map.setLayoutProperty("hillshade", "visibility", view3d ? "visible" : "none");
    map.easeTo({ pitch: view3d ? 55 : 0, duration: 800 });
  }, [ready, view3d]);

  const flyTo = (b: Bounds, maxZoom = 12) =>
    mapRef.current?.fitBounds([[b[0], b[1]], [b[2], b[3]]], {
      // Leave room on the left for the chip card.
      padding: { top: 60, bottom: 60, right: 60, left: window.innerWidth > 1100 ? 360 : 60 },
      maxZoom,
      duration: 900,
    });

  const selectChip = (f: ChipFeature) => {
    setSelectedId(f.id);
    flyTo(boundsOf(f), 13);
  };

  const openEvent = (e: FloodEvent) => {
    setActiveEvent(e.id);
    setSceneIndex(0);
    setScenes([]);
    flyTo(e.bbox, 9);
    getJson<{ scenes: Scene[] }>(`/api/events/${e.id}/scenes`)
      .then((r) => setScenes(r.scenes))
      .catch((err) => setError(err.message));
  };

  const ask = (text: string) => {
    const q = text.trim();
    if (q.length < 3 || asking) return;
    setQuestion(q);
    setAsking(true);
    setReport(null);
    setError(null);
    getJson<AskReport>(`/api/ask?q=${encodeURIComponent(q)}`)
      .then((r) => {
        setReport(r);
        const boxes = [...r.regions.map((x) => x.bbox)];
        if (boxes.length) {
          flyTo(
            [
              Math.min(...boxes.map((b) => b[0])),
              Math.min(...boxes.map((b) => b[1])),
              Math.max(...boxes.map((b) => b[2])),
              Math.max(...boxes.map((b) => b[3])),
            ],
            9,
          );
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setAsking(false));
  };

  const eventRows = useMemo(() => {
    const byEvent = new Map<string, { chips: number; ious: number[]; weights: number[]; bounds: Bounds }>();
    for (const row of summary) {
      if (split !== "all" && row.split !== split) continue;
      const cur = byEvent.get(row.event) ?? {
        chips: 0,
        ious: [],
        weights: [],
        bounds: [...row.bounds] as Bounds,
      };
      cur.chips += row.chips;
      if (row.mean_chip_iou !== null) {
        cur.ious.push(row.mean_chip_iou);
        cur.weights.push(row.chips);
      }
      cur.bounds = [
        Math.min(cur.bounds[0], row.bounds[0]),
        Math.min(cur.bounds[1], row.bounds[1]),
        Math.max(cur.bounds[2], row.bounds[2]),
        Math.max(cur.bounds[3], row.bounds[3]),
      ];
      byEvent.set(row.event, cur);
    }
    return [...byEvent.entries()].map(([event, v]) => {
      const total = v.weights.reduce((a, b) => a + b, 0);
      const iou = total ? v.ious.reduce((a, x, i) => a + x * v.weights[i], 0) / total : null;
      return { event, chips: v.chips, iou, bounds: v.bounds };
    });
  }, [summary, split]);

  const chipList = useMemo(
    () =>
      eventFilter
        ? [...visible].sort((a, b) => (a.properties.iou ?? 2) - (b.properties.iou ?? 2))
        : [],
    [visible, eventFilter],
  );

  const scene = scenes[sceneIndex];

  return (
    <div className="shell">
      <aside className="panel">
        <header>
          <p className="eyebrow">TERRA MIND</p>
          <h1>Earth Explorer</h1>
          {chips && (
            <p className="muted">
              Model: {chips.model.name}, {chips.model.seeds.length} seeds
            </p>
          )}
          <div className="pills view-toggle">
            <button aria-pressed={!view3d} onClick={() => setView3d(false)}>
              2D map
            </button>
            <button aria-pressed={view3d} onClick={() => setView3d(true)}>
              3D globe
            </button>
          </div>
        </header>

        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === "benchmark"} onClick={() => setTab("benchmark")}>
            Benchmark
          </button>
          <button role="tab" aria-selected={tab === "events"} onClick={() => setTab("events")}>
            Events
          </button>
          <button role="tab" aria-selected={tab === "ask"} onClick={() => setTab("ask")}>
            Ask
          </button>
          <button role="tab" aria-selected={tab === "simulate"} onClick={() => setTab("simulate")}>
            Simulate
          </button>
        </div>

        {error && <p className="error">{error}</p>}

        {tab === "benchmark" && (
          <>
            <section>
              <h2>Split</h2>
              <div className="pills">
                {SPLITS.map((s) => (
                  <button key={s} aria-pressed={split === s} onClick={() => setSplit(s)}>
                    {s === "bolivia" ? "Bolivia (unseen)" : s}
                  </button>
                ))}
              </div>
            </section>

            <section>
              <h2>
                Flood events <span className="muted">mean chip IoU</span>
              </h2>
              <ul className="rows">
                {eventRows.map((r) => (
                  <li key={r.event}>
                    <button
                      aria-pressed={eventFilter === r.event}
                      onClick={() => {
                        setEventFilter(eventFilter === r.event ? null : r.event);
                        flyTo(r.bounds, 9);
                      }}
                    >
                      <span className="name">{r.event}</span>
                      <span className="count">{r.chips}</span>
                      <span className="bar">
                        <i style={{ width: `${(r.iou ?? 0) * 100}%` }} />
                      </span>
                      <span className="num">{fmt(r.iou)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>

            {eventFilter && (
              <section>
                <h2>
                  {eventFilter} chips <span className="muted">worst first</span>
                </h2>
                <ul className="rows chips">
                  {chipList.map((f) => (
                    <li key={f.id}>
                      <button aria-pressed={selectedId === f.id} onClick={() => selectChip(f)}>
                        <span className="name">{f.id.split("_")[1]}</span>
                        <span className="count">{f.properties.split}</span>
                        <span className="num">{fmt(f.properties.iou)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </>
        )}

        {tab === "events" && (
          <section>
            <h2>Events for scene discovery</h2>
            <ul className="rows">
              {events.map((e) => (
                <li key={e.id}>
                  <button aria-pressed={activeEvent === e.id} onClick={() => openEvent(e)}>
                    <span className="name">{e.name}</span>
                    <span className="count">{e.start.slice(0, 7)}</span>
                  </button>
                </li>
              ))}
            </ul>
            {activeEvent && (
              <div className="timeline">
                {scenes.length === 0 ? (
                  <p className="muted">No saved scene search for this event.</p>
                ) : (
                  <>
                    <label htmlFor="scene">
                      Scene {sceneIndex + 1} of {scenes.length}
                    </label>
                    <input
                      id="scene"
                      type="range"
                      min={0}
                      max={scenes.length - 1}
                      value={sceneIndex}
                      onChange={(e) => setSceneIndex(Number(e.target.value))}
                    />
                    {scene && (
                      <dl>
                        <dt>Acquired</dt>
                        <dd>{scene.datetime?.slice(0, 16).replace("T", " ")} UTC</dd>
                        <dt>Collection</dt>
                        <dd>{scene.collection}</dd>
                        <dt>Platform</dt>
                        <dd>{scene.platform ?? "n/a"}</dd>
                        <dt>Cloud cover</dt>
                        <dd>{scene.cloud_cover === null ? "n/a (radar)" : `${scene.cloud_cover.toFixed(0)}%`}</dd>
                      </dl>
                    )}
                    <p className="muted">
                      Event dates and boxes are approximate and unconfirmed. Footprints only; imagery is not
                      downloaded.
                    </p>
                  </>
                )}
              </div>
            )}
          </section>
        )}
        {tab === "simulate" && (
          <section className="sim">
            <h2>Flood response simulation</h2>
            <p className="warn">
              Simulated. The flood extent and the model prediction are real; the teams and what counts
              as reached are invented.
            </p>
            {!selected ? (
              <p className="muted">Zoom in and click a chip on the map to choose a scenario.</p>
            ) : (
              <>
                <p>
                  {selected.properties.event} chip {selected.id.split("_")[1]} · three teams start at the
                  top-left corner and have 48 moves.
                </p>
                <h4>Policy</h4>
                <div className="pills">
                  {POLICIES.map(([id, label]) => (
                    <button
                      key={id}
                      aria-pressed={policy === id && episode?.chip === selected.id}
                      onClick={() => simulate(selected.id, id)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                {episode && episode.chip === selected.id && (
                  <div className="timeline">
                    <label htmlFor="simstep">
                      Step {simStep} of {episode.horizon}
                    </label>
                    <input
                      id="simstep"
                      type="range"
                      min={0}
                      max={episode.horizon}
                      value={simStep}
                      onChange={(e) => {
                        setPlaying(false);
                        setSimStep(Number(e.target.value));
                      }}
                    />
                    <button
                      className="primary"
                      onClick={() => {
                        if (simStep >= episode.horizon) setSimStep(0);
                        setPlaying(!playing);
                      }}
                    >
                      {playing ? "Pause" : simStep >= episode.horizon ? "Replay" : "Play"}
                    </button>
                    <dl>
                      <dt>Flooded area reached so far</dt>
                      <dd>{pct(episode.served[simStep])}</dd>
                      <dt>Reached by the end</dt>
                      <dd>{pct(episode.served[episode.horizon])}</dd>
                      <dt>Average over the episode</dt>
                      <dd>{pct(episode.auc)}</dd>
                    </dl>
                    <h4>Blue cells show</h4>
                    <div className="pills">
                      <button aria-pressed={simShow === "truth"} onClick={() => setSimShow("truth")}>
                        Labelled water
                      </button>
                      <button
                        aria-pressed={simShow === "prediction"}
                        onClick={() => setSimShow("prediction")}
                      >
                        Model prediction
                      </button>
                    </div>
                    <p className="muted">
                      Green cells have been visited. Teams see only the model prediction until they are
                      next to a cell.
                    </p>
                  </div>
                )}
              </>
            )}
          </section>
        )}
        {tab === "ask" && (
          <section className="ask">
            <h2>Ask TerraCore</h2>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                ask(question);
              }}
            >
              <textarea
                aria-label="Question"
                rows={3}
                value={question}
                placeholder="Ask about model performance, scenes, chips or datasets"
                onChange={(e) => setQuestion(e.target.value)}
              />
              <button type="submit" className="primary" disabled={asking || question.trim().length < 3}>
                {asking ? "Checking evidence…" : "Ask"}
              </button>
            </form>
            {!report && !asking && (
              <ul className="examples">
                {EXAMPLES.map((x) => (
                  <li key={x}>
                    <button onClick={() => ask(x)}>{x}</button>
                  </li>
                ))}
              </ul>
            )}
            {report && (
              <div className="answer">
                <p className={report.verdict === "abstain" ? "verdict abstain" : "verdict"}>
                  {report.verdict === "abstain" ? "Abstained" : "Answered from evidence"}
                </p>
                {report.answer.split("\n").map((line, i) =>
                  line.trim() ? <p key={i}>{line.replace(/^- /, "")}</p> : null,
                )}
                {report.caveats.length > 0 && (
                  <>
                    <h4>Caveats</h4>
                    <ul className="caveats">
                      {report.caveats.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </>
                )}
                {report.evidence.length > 0 && (
                  <>
                    <h4>Evidence</h4>
                    <ol className="evidence">
                      {report.evidence.map((e) => (
                        <li key={e.id}>
                          <span>{e.title}</span>
                          <code>{e.id}</code>
                          <span className="muted">
                            {e.kind} · provenance {e.provenance.toFixed(1)}
                            {e.start ? ` · ${e.start}` : ""}
                          </span>
                        </li>
                      ))}
                    </ol>
                  </>
                )}
                <details>
                  <summary>Agent trace ({report.trace.length} steps)</summary>
                  <ol className="trace">
                    {report.trace.map((t, i) => (
                      <li key={i}>
                        <b>
                          {t.agent}.{t.action}
                        </b>
                        <code>{JSON.stringify(t.detail)}</code>
                      </li>
                    ))}
                  </ol>
                </details>
                <p className="muted">
                  Answers are assembled from stored evidence by fixed rules; no language model writes
                  them. Forecasting is not built yet.
                </p>
              </div>
            )}
          </section>
        )}
      </aside>

      <main className="stage">
        <div ref={container} className="map" />

        {tab === "benchmark" && selected && (
          <div className="card">
            <div className="card-head">
              <div>
                <p className="eyebrow">
                  {selected.properties.event} · {selected.properties.split}
                </p>
                <h3>Chip {selected.id.split("_")[1]}</h3>
              </div>
              <button className="close" aria-label="Close chip" onClick={() => setSelectedId(null)}>
                ×
              </button>
            </div>
            {selected.properties.split === "train" && (
              <p className="warn">Seen during training. These scores are in-sample.</p>
            )}
            <dl className="stats">
              <dt>Water IoU</dt>
              <dd>{fmt(selected.properties.iou, 3)}</dd>
              <dt>Precision</dt>
              <dd>{fmt(selected.properties.precision, 3)}</dd>
              <dt>Recall</dt>
              <dd>{fmt(selected.properties.recall, 3)}</dd>
              <dt>Labelled water</dt>
              <dd>{pct(selected.properties.water_fraction)}</dd>
              <dt>Mean disagreement</dt>
              <dd>{fmt(selected.properties.mean_uncertainty, 3)}</dd>
            </dl>

            <h4>Imagery</h4>
            <div className="pills">
              {BASES.map((b) => (
                <button key={b} aria-pressed={base === b} onClick={() => setBase(b)}>
                  {b === "sar" ? "Radar (VV)" : b === "optical" ? "Optical" : "None"}
                </button>
              ))}
            </div>

            <h4>Overlays</h4>
            {OVERLAYS.map((o) => (
              <label key={o} className="check">
                <input
                  type="checkbox"
                  checked={overlays.includes(o)}
                  onChange={(e) =>
                    setOverlays(e.target.checked ? [...overlays, o] : overlays.filter((x) => x !== o))
                  }
                />
                <span>{OVERLAY_INFO[o].label}</span>
                <span className="swatches">
                  {OVERLAY_INFO[o].swatches.map(([colour, name]) => (
                    <span key={name}>
                      <i style={{ background: colour }} />
                      {name}
                    </span>
                  ))}
                </span>
              </label>
            ))}
            <h4>Situation report</h4>
            {situation?.chip === selected.id ? (
              <div className="situation">
                <p className={situation.abstain ? "verdict abstain" : "verdict"}>
                  Reliability {situation.reliability}
                </p>
                <p>{situation.recommendation}</p>
                {situation.summary?.accepted && (
                  <p className="summary">{situation.summary.text}</p>
                )}
                {situation.summary && !situation.summary.accepted && (
                  <p className="muted">Summary not shown: {situation.summary.reason}.</p>
                )}
                {!situation.summary && (
                  <button
                    className="link"
                    disabled={loadingSituation}
                    onClick={() => loadSituation(selected.id, true)}
                  >
                    {loadingSituation ? "Writing…" : "Plain-language summary (local LLM)"}
                  </button>
                )}
                {situation.flags.length > 0 && (
                  <ul className="caveats">
                    {situation.flags.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                )}
                {situation.sections.map((sec) => (
                  <details key={sec.title}>
                    <summary>{sec.title}</summary>
                    <ul className="evidence">
                      {sec.statements.map((st) => (
                        <li key={st.text}>
                          <span>
                            {st.simulated && <b>Simulated. </b>}
                            {st.text.replace(/^Simulated: /, "")}
                          </span>
                          <code>{st.evidence.join(", ")}</code>
                        </li>
                      ))}
                    </ul>
                  </details>
                ))}
                <p className="muted">{situation.caveats.join(" ")}</p>
              </div>
            ) : (
              <button
                className="primary"
                disabled={loadingSituation}
                onClick={() => loadSituation(selected.id)}
              >
                {loadingSituation ? "Building…" : "Build report"}
              </button>
            )}
            <label className="slider">
              Overlay opacity
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
            </label>
          </div>
        )}

        {tab === "benchmark" && (
          <div className="legend">
            <span>Chip IoU</span>
            <i className="ramp" />
            <span className="ends">
              <b>0</b>
              <b>1</b>
            </span>
          </div>
        )}
      </main>
    </div>
  );
}
