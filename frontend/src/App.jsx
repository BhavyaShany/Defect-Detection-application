import React, { useCallback, useEffect, useRef, useState } from "react";
import ThreeViewer from "./ThreeViewer";
import { API_BASE, ApiError, decodeGeometry, getJson, uploadScan } from "./api";
import { useLiveFeed } from "./useLiveFeed";
import {
  AnalyticsView, HistoryTable, KpiRow, ModelBanner, PipelineStatus, ResultPanel, TimingPanel,
} from "./components";
import "./App.css";

const PAGE_SIZE = 15;
const DEFAULT_EXT = [".ply", ".pcd", ".stl", ".obj"];
const IDLE = { phase: "idle" };

export default function App() {
  const [system, setSystem] = useState(null);
  const [summary, setSummary] = useState(null);
  const [pareto, setPareto] = useState([]);
  const [shifts, setShifts] = useState(null);
  const [activity, setActivity] = useState([]);
  const [performance, setPerformance] = useState(null);

  const [history, setHistory] = useState({ items: [], total: 0 });
  const [skip, setSkip] = useState(0);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [histLoading, setHistLoading] = useState(false);
  const [histError, setHistError] = useState(null);

  const [current, setCurrent] = useState(null);
  const [geometry, setGeometry] = useState(null);
  const [geoLoading, setGeoLoading] = useState(false);
  const [geoError, setGeoError] = useState(null);
  const [clientTiming, setClientTiming] = useState({});
  const [loadError, setLoadError] = useState(null);

  const [upload, setUpload] = useState(IDLE);
  const [meta, setMeta] = useState({ part_type: "", batch_id: "", shift: "", operator_id: "" });
  const [tab, setTab] = useState("3d");
  const [follow, setFollow] = useState(true);
  const [dragOver, setDragOver] = useState(false);

  const uploadRef = useRef(IDLE);
  const selectToken = useRef(0);
  const refreshTimer = useRef(null);
  const busy = upload.phase === "uploading" || upload.phase === "processing";

  const updateUpload = useCallback((next) => {
    uploadRef.current = typeof next === "function" ? next(uploadRef.current) : next;
    setUpload(uploadRef.current);
  }, []);

  // ---------------- data loading ----------------
  const refreshAnalytics = useCallback(async () => {
    const [s, p, sh, a, perf] = await Promise.allSettled([
      getJson("/api/metrics/summary"),
      getJson("/api/metrics/pareto"),
      getJson("/api/metrics/shifts"),
      getJson("/api/metrics/activity?hours=24"),
      getJson("/api/metrics/performance"),
    ]);
    if (s.status === "fulfilled") setSummary(s.value);
    if (p.status === "fulfilled") setPareto(p.value);
    if (sh.status === "fulfilled") setShifts(sh.value);
    if (a.status === "fulfilled") setActivity(a.value);
    if (perf.status === "fulfilled") setPerformance(perf.value);
  }, []);

  const refreshHistory = useCallback(async () => {
    setHistLoading(true);
    try {
      const q = new URLSearchParams({ skip: String(skip), limit: String(PAGE_SIZE) });
      if (filter !== "all") q.set("status", filter);
      if (debouncedSearch.trim()) q.set("search", debouncedSearch.trim());
      setHistory(await getJson(`/api/scans?${q}`));
      setHistError(null);
    } catch (e) {
      setHistError(e.message);
    } finally {
      setHistLoading(false);
    }
  }, [skip, filter, debouncedSearch]);

  const refreshSystem = useCallback(async () => {
    try {
      setSystem(await getJson("/api/system/status"));
    } catch (e) {
      setSystem(null);
    }
  }, []);

  const scheduleRefresh = useCallback(() => {
    clearTimeout(refreshTimer.current);
    refreshTimer.current = setTimeout(() => {
      refreshAnalytics();
      refreshHistory();
    }, 300);
  }, [refreshAnalytics, refreshHistory]);

  useEffect(() => { refreshSystem(); refreshAnalytics(); }, [refreshSystem, refreshAnalytics]);
  useEffect(() => { refreshHistory(); }, [refreshHistory]);
  useEffect(() => {
    const t = setTimeout(() => { setDebouncedSearch(search); setSkip(0); }, 350);
    return () => clearTimeout(t);
  }, [search]);
  useEffect(() => { setSkip(0); }, [filter]);
  useEffect(() => () => clearTimeout(refreshTimer.current), []);

  // ---------------- selecting a scan ----------------
  const loadGeometry = useCallback(async (id, token) => {
    setGeoLoading(true);
    setGeoError(null);
    const t0 = performance_now();
    try {
      const payload = await getJson(`/api/scans/${id}/geometry`, { timeoutMs: 60000 });
      const decoded = decodeGeometry(payload);
      if (token !== selectToken.current) return;
      setGeometry(decoded);
      setClientTiming((c) => ({ ...c, geometryLoadMs: performance_now() - t0 }));
    } catch (e) {
      if (token !== selectToken.current) return;
      setGeometry(null);
      setGeoError(e.message);
    } finally {
      if (token === selectToken.current) setGeoLoading(false);
    }
  }, []);

  const selectScan = useCallback(async (id, detail, timing) => {
    const token = ++selectToken.current;
    setLoadError(null);
    setClientTiming(timing || {});
    setGeometry(null);
    let rec = detail;
    if (!rec) {
      try {
        rec = await getJson(`/api/scans/${id}`);
      } catch (e) {
        if (token === selectToken.current) setLoadError(e.message);
        return;
      }
    }
    if (token !== selectToken.current) return;
    setCurrent(rec);
    if (rec.input_type === "image") {  // photographs have no 3D geometry
      setGeometry(null);
      setGeoError(null);
      setGeoLoading(false);
    } else {
      loadGeometry(id, token);
    }
  }, [loadGeometry]);

  // ---------------- live events ----------------
  const handleEvent = useCallback((ev) => {
    const u = uploadRef.current;
    const ours = (u.phase === "uploading" || u.phase === "processing");
    if (ev.type === "progress" && ours && ev.filename === u.filename) {
      updateUpload((p) => ({ ...p, phase: "processing", stage: ev.stage, message: ev.message }));
    } else if (ev.type === "result") {
      scheduleRefresh();
      if (!ours && follow && ev.scan_id) selectScan(ev.scan_id);
    } else if (ev.type === "reconnected") {
      refreshSystem();
      scheduleRefresh();
    }
  }, [follow, scheduleRefresh, selectScan, refreshSystem, updateUpload]);

  const wsStatus = useLiveFeed(handleEvent);

  // Polling fallback only while the socket is down
  useEffect(() => {
    if (wsStatus === "open") return undefined;
    const t = setInterval(scheduleRefresh, 10000);
    return () => clearInterval(t);
  }, [wsStatus, scheduleRefresh]);

  // ---------------- upload ----------------
  const handleFile = useCallback(async (file) => {
    if (!file || uploadRef.current.phase === "uploading" || uploadRef.current.phase === "processing") return;
    const allowed = system?.limits?.allowed_extensions || DEFAULT_EXT;
    const maxMb = system?.limits?.max_upload_mb;
    const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
    const fail = (msg) => updateUpload({ phase: "error", filename: file.name, error: msg });

    if (!allowed.includes(ext)) return fail(`Unsupported file type "${ext || file.name}". Allowed: ${allowed.join(", ")}.`);
    if (file.size === 0) return fail("The selected file is empty.");
    if (maxMb && file.size > maxMb * 1024 * 1024) return fail(`The file is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${maxMb} MB.`);

    updateUpload({ phase: "uploading", filename: file.name, percent: 0, stage: "upload" });
    try {
      const { data, uploadMs, roundTripMs } = await uploadScan(file, meta, {
        onUploadProgress: (pct) => updateUpload((u) => (u.phase === "uploading" ? { ...u, percent: pct } : u)),
        onUploadDone: () => updateUpload((u) => ({ ...u, phase: "processing", stage: "parsing", message: "Server is processing the scan…" })),
      });
      updateUpload({ phase: "done", filename: file.name });
      await selectScan(data.id, data, { uploadMs, roundTripMs });
      scheduleRefresh();
    } catch (e) {
      updateUpload({ phase: "error", filename: file.name, error: e instanceof ApiError ? e.message : "Upload failed." });
    }
  }, [system, meta, selectScan, scheduleRefresh, updateUpload]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) handleFile(e.dataTransfer.files[0]);
  };

  const wsLabel = { open: "Live updates connected", connecting: "Connecting…", reconnecting: "Connection lost — reconnecting (polling)" }[wsStatus];

  return (
    <div className="app-container">
      <header className="main-header">
        <div className="header-brand">
          <div className="brand-icon">⚙️</div>
          <div>
            <h1>Cast Metal 3D Defect Inspection System</h1>
            <div className="header-meta">
              <span className="badge-version">v{system?.version || "…"}</span>
              <span className={`ws-status ${wsStatus === "open" ? "connected" : "disconnected"}`}>● {wsLabel}</span>
              {system && <span className={`ws-status ${system.model.ready ? "connected" : "disconnected"}`}>● Model: {system.model.ready ? system.model.model_name : "not loaded"}</span>}
              {!system && <span className="ws-status disconnected">● Backend unreachable ({API_BASE || "this server"})</span>}
            </div>
          </div>
        </div>
        <div className="header-actions">
          <label className="inline-check" title="Automatically show scans uploaded by other stations">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow live scans
          </label>
          <div className="dropdown">
            <button className="btn btn-secondary">📊 Export ▾</button>
            <div className="dropdown-menu">
              <button onClick={() => window.open(`${API_BASE}/api/metrics/export?format=csv`, "_blank")}>CSV report</button>
              <button onClick={() => window.open(`${API_BASE}/api/metrics/export?format=json`, "_blank")}>JSON data</button>
            </div>
          </div>
        </div>
      </header>

      <ModelBanner model={system?.model} />
      <KpiRow summary={summary} current={current} performance={performance} />

      <div className="workspace-grid">
        <div className="stage-card">
          <div className="card-header">
            <div className="tabs">
              <button className={`tab-btn ${tab === "3d" ? "active" : ""}`} onClick={() => setTab("3d")}>{current?.input_type === "image" ? "🖼 Photo Inspection" : "🎮 3D Inspection"}</button>
              <button className={`tab-btn ${tab === "maps" ? "active" : ""}`} onClick={() => setTab("maps")}>👁 Depth &amp; Normal Maps</button>
              <button className={`tab-btn ${tab === "stats" ? "active" : ""}`} onClick={() => setTab("stats")}>📈 Analytics</button>
            </div>
            {current && <span className="part-tag">{current.part_id}</span>}
          </div>
          <div className="card-body">
            {loadError && <div className="banner banner-warning small">{loadError}</div>}
            {tab === "3d" && current?.input_type === "image" && (
              <div className="viewer-container image-inspect">
                {current.image_base64 ? <img src={current.image_base64} alt="Inspected part" /> : <div className="vision-placeholder">No preview stored</div>}
                <div className="image-caption">
                  Photograph inspection: the model classifies the whole image and does not report where a defect is.
                </div>
              </div>
            )}
            {tab === "3d" && current?.input_type !== "image" && (
              <div className="viewer-container">
                <ThreeViewer geometry={geometry} loading={geoLoading} error={geoError} />
              </div>
            )}
            {tab === "maps" && current?.input_type === "image" && (
              <p className="text-muted">Depth and normal maps exist only for 3D scans. This record is a photograph.</p>
            )}
            {tab === "maps" && current?.input_type !== "image" && (
              <div className="vision-view two-col">
                <div className="vision-column">
                  <div className="vision-label">Orthographic depth map (CNN input)</div>
                  {current?.depth_map_base64 ? <img className="vision-img" src={current.depth_map_base64} alt="Depth map" /> : <div className="vision-placeholder">No scan selected</div>}
                  <span className="vision-caption">Top-down z-buffer, scaled by a fixed physical reference. Black = no data.</span>
                </div>
                <div className="vision-column">
                  <div className="vision-label">Surface-normal map</div>
                  {current?.normal_map_base64 ? <img className="vision-img" src={current.normal_map_base64} alt="Normal map" /> : <div className="vision-placeholder">No scan selected</div>}
                  <span className="vision-caption">Normals derived from the depth map.</span>
                </div>
              </div>
            )}
            {tab === "stats" && <AnalyticsView pareto={pareto} shifts={shifts} activity={activity} performance={performance} />}
          </div>
        </div>

        <div className="control-col">
          <div className="stage-card">
            <div className="card-header"><h2>NEW INSPECTION</h2></div>
            <div className="card-body">
              <div
                className={`dropzone ${dragOver ? "over" : ""} ${busy ? "disabled" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={busy ? (e) => e.preventDefault() : onDrop}
              >
                <label className={`btn btn-primary ${busy ? "btn-disabled" : ""}`}>
                  {busy ? "Inspection in progress…" : system?.limits?.image_inspection ? "📁 Choose scan or photo" : "📁 Choose 3D scan"}
                  <input type="file" hidden disabled={busy} accept={(system?.limits?.allowed_extensions || DEFAULT_EXT).join(",")}
                    onChange={(e) => { handleFile(e.target.files[0]); e.target.value = ""; }} />
                </label>
                <div className="text-muted small">or drop a file here · {(system?.limits?.allowed_extensions || DEFAULT_EXT).join(" ")}{system?.limits ? ` · max ${system.limits.max_upload_mb} MB` : ""}</div>
              </div>
              <div className="meta-form">
                {[["part_type", "Part type"], ["batch_id", "Batch ID"], ["shift", "Shift"], ["operator_id", "Operator"]].map(([k, label]) => (
                  <label key={k}>{label}
                    <input className="select-input" value={meta[k]} maxLength={64} placeholder="optional" disabled={busy}
                      onChange={(e) => setMeta((m) => ({ ...m, [k]: e.target.value }))} />
                  </label>
                ))}
              </div>
              <PipelineStatus upload={upload} />
            </div>
          </div>

          <div className="stage-card decision-panel">
            <div className="card-header"><h2>INSPECTION RESULT</h2></div>
            <div className="card-body"><ResultPanel scan={current} /></div>
          </div>

          {current && (
            <div className="stage-card">
              <div className="card-header"><h2>PROCESSING TIME (MEASURED)</h2></div>
              <div className="card-body"><TimingPanel timings={current.timings} client={clientTiming} /></div>
            </div>
          )}
        </div>
      </div>

      <HistoryTable
        items={history.items} total={history.total} skip={skip} limit={PAGE_SIZE}
        filter={filter} setFilter={setFilter} search={search} setSearch={setSearch}
        onPage={setSkip} onSelect={(id) => selectScan(id)} currentId={current?.id}
        loading={histLoading} error={histError}
      />
    </div>
  );
}

function performance_now() {
  return window.performance.now();
}
