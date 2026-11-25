import React from "react";

// ---------- formatters (null/undefined always render as an em dash, never a made-up number) ----------
export const fmtPct = (v, digits = 1) => (v == null ? "—" : `${Number(v).toFixed(digits)}%`);
export const fmtMs = (v) => (v == null ? "—" : `${Number(v).toFixed(v >= 100 ? 0 : 1)} ms`);
export const fmtConf = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
export const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : "—");
const fmtBytes = (n) => (n == null ? "—" : n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${(n / 1024).toFixed(0)} KB`);

const STATUS_LABEL = { PASS: "PASS", FAIL: "FAIL", REVIEW: "REVIEW", UNCLASSIFIED: "UNCLASSIFIED" };

export function StatusPill({ status }) {
  return <span className={`status-pill status-${(status || "").toLowerCase()}`}>{STATUS_LABEL[status] || status}</span>;
}

// ---------- model banner ----------
export function ModelBanner({ model }) {
  if (model?.ready && model.input_mode === "image" && model.ood && !model.ood.enabled) {
    return (
      <div className="banner banner-warning">
        <strong>Input check not active.</strong> Photos that are not castings (people, other objects) can still receive a
        confident PASS/FAIL. Run <code>python -m ml.fit_ood</code> and restart the server.
        <span className="banner-detail">{model.ood.error}</span>
      </div>
    );
  }
  if (!model || model.ready) return null;
  return (
    <div className="banner banner-warning">
      <strong>No trained model loaded.</strong> Scans are still preprocessed and shown in 3D, but they cannot be classified and are stored as
      UNCLASSIFIED. <span className="banner-detail">{model.error}</span>
    </div>
  );
}

// ---------- KPI cards ----------
export function KpiRow({ summary, current, performance }) {
  const s = summary || {};
  const p95 = performance?.stages_ms?.server_total_ms?.p95;
  return (
    <section className="metrics-row">
      <div className="metric-card">
        <span className="metric-title">TOTAL INSPECTED</span>
        <span className="metric-value">{s.total_scanned ?? "—"}</span>
        <span className="metric-sub">
          {s.classified != null ? `${s.classified} classified · ${s.unclassified} unclassified` : "No data yet"}
        </span>
      </div>
      <div className="metric-card">
        <span className="metric-title">PASS RATE</span>
        <span className="metric-value text-success">{fmtPct(s.pass_rate_pct)}</span>
        <div className="progress-bar-bg">
          <div className="progress-bar-fill fill-pass" style={{ width: `${s.pass_rate_pct ?? 0}%` }} />
        </div>
      </div>
      <div className="metric-card">
        <span className="metric-title">DEFECT RATE</span>
        <span className="metric-value text-danger">{fmtPct(s.defect_rate_pct)}</span>
        <div className="progress-bar-bg">
          <div className="progress-bar-fill fill-fail" style={{ width: `${s.defect_rate_pct ?? 0}%` }} />
        </div>
      </div>
      <div className="metric-card">
        <span className="metric-title">NEEDS REVIEW</span>
        <span className="metric-value text-warn">{s.needs_review ?? "—"}</span>
        <span className="metric-sub">Below confidence threshold</span>
      </div>
      <div className="metric-card">
        <span className="metric-title">AVG PROCESSING</span>
        <span className="metric-value text-info">{fmtMs(s.avg_latency_ms)}</span>
        <span className="metric-sub">{p95 != null ? `p95 ${fmtMs(p95)} (measured)` : "Server-side, measured"}</span>
      </div>
      <div className="metric-card">
        <span className="metric-title">LAST CONFIDENCE</span>
        <span className="metric-value text-highlight">{fmtConf(current?.confidence)}</span>
        <span className="metric-sub">{current?.model_name || "Selected scan"}</span>
      </div>
    </section>
  );
}

// ---------- upload / pipeline status ----------
const STAGES = [
  ["upload", "Upload"],
  ["parsing", "Parse"],
  ["preprocessing", "Preprocess"],
  ["inference", "Inference"],
  ["saving", "Save"],
];

export function PipelineStatus({ upload }) {
  if (upload.phase === "idle") return null;
  const order = STAGES.map((s) => s[0]);
  const activeIdx = upload.phase === "done" ? order.length : Math.max(0, order.indexOf(upload.stage || "upload"));
  return (
    <div className={`pipeline-card ${upload.phase === "error" ? "pipeline-error" : ""}`}>
      <div className="pipeline-head">
        <strong>{upload.filename}</strong>
        <span className="text-muted">
          {upload.phase === "uploading" && `Uploading… ${upload.percent}%`}
          {upload.phase === "processing" && (upload.message || "Processing…")}
          {upload.phase === "done" && "Inspection complete"}
          {upload.phase === "error" && "Failed"}
        </span>
      </div>
      {upload.phase === "uploading" && (
        <div className="progress-bar-bg">
          <div className="progress-bar-fill fill-info" style={{ width: `${upload.percent}%` }} />
        </div>
      )}
      {upload.phase !== "error" && (
        <div className="stage-steps">
          {STAGES.map(([id, label], i) => (
            <span key={id} className={`stage-step ${i < activeIdx ? "done" : i === activeIdx ? "active" : ""}`}>
              {label}
            </span>
          ))}
        </div>
      )}
      {upload.phase === "error" && <div className="pipeline-errmsg">{upload.error}</div>}
    </div>
  );
}

// ---------- decision / result ----------
const VERDICT = {
  PASS: { cls: "verdict-pass", icon: "✓", title: "PASS — NO DEFECT PREDICTED" },
  FAIL: { cls: "verdict-fail", icon: "⚠", title: "FAIL — DEFECT PREDICTED" },
  REVIEW: { cls: "verdict-review", icon: "?", title: "MANUAL REVIEW — LOW CONFIDENCE" },
  UNCLASSIFIED: { cls: "verdict-unclassified", icon: "–", title: "NOT CLASSIFIED" },
};

export function ResultPanel({ scan }) {
  if (!scan) {
    return (
      <div className="empty-state">
        <p>No scan selected.</p>
        <p className="text-muted">Upload a scan or photo, or select one from the inspection log.</p>
      </div>
    );
  }
  const v = VERDICT[scan.status] || VERDICT.UNCLASSIFIED;
  const m = scan.metrics || {};
  const sv = m.surface_variation || {};
  const probs = scan.probabilities || {};
  return (
    <>
      <div className={`verdict-banner ${v.cls}`}>
        <div className="verdict-icon">{v.icon}</div>
        <div>
          <div className="verdict-title">{v.title}</div>
          <div className="verdict-sub">
            {scan.defect_type ? (
              <>
                Predicted class: <strong>{scan.defect_type}</strong> · confidence <strong>{fmtConf(scan.confidence)}</strong>
              </>
            ) : scan.metrics?.ood?.flagged ? (
              "Not recognised as a casting photo - not classified"
            ) : (
              "No model result available for this scan"
            )}
          </div>
        </div>
      </div>

      <div className="meta-grid">
        <div><span className="meta-lbl">{scan.input_type === "image" ? "Image / Part ID" : "Scan / Part ID"}</span><span className="meta-val">{scan.part_id}</span></div>
        <div><span className="meta-lbl">File</span><span className="meta-val" title={scan.filename}>{scan.filename || "—"}</span></div>
        <div><span className="meta-lbl">Timestamp</span><span className="meta-val">{fmtTime(scan.timestamp)}</span></div>
        <div><span className="meta-lbl">Processing</span><span className="meta-val">{fmtMs(scan.latency_ms)}</span></div>
        <div><span className="meta-lbl">Batch</span><span className="meta-val">{scan.batch_id}</span></div>
        <div><span className="meta-lbl">Shift / Operator</span><span className="meta-val">{scan.shift} · {scan.operator_id}</span></div>
        <div><span className="meta-lbl">Model</span><span className="meta-val">{scan.model_name ? `${scan.model_name} (${scan.model_version})` : "—"}</span></div>
        <div><span className="meta-lbl">File size</span><span className="meta-val">{fmtBytes(scan.file_size_bytes)}</span></div>
      </div>

      {scan.warnings?.length > 0 && (
        <div className="banner banner-warning small">
          {scan.warnings.map((w, i) => (<div key={i}>• {w}</div>))}
        </div>
      )}

      {Object.keys(probs).length > 0 && (
        <div className="probs-container">
          <div className="box-title">Class probabilities</div>
          {Object.entries(probs).sort((a, b) => b[1] - a[1]).map(([cls, p]) => (
            <div key={cls} className="prob-item">
              <span className="prob-label">{cls}</span>
              <div className="prob-bar-bg">
                <div className={`prob-bar-fill ${cls === scan.defect_type ? "active" : ""}`} style={{ width: `${Math.round(p * 100)}%` }} />
              </div>
              <span className="prob-num">{(p * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {scan.input_type === "image" ? (
        <div className="open3d-metrics-box">
          <div className="box-title">Photograph</div>
          <div className="stats-mini-row">
            <span>Size: <strong>{m.width && m.height ? `${m.width} × ${m.height} px` : "—"}</strong></span>
            <span>Format: <strong>{m.input_format || "—"}</strong></span>
            <span>Colour mode: <strong>{m.color_mode || "—"}</strong></span>
          </div>
          {m.ood && (
            <div className="stats-mini-row">
              <span>Input check: <strong>{m.ood.flagged ? "not a casting photo" : "looks like training photos"}</strong></span>
              <span>Distance: <strong>{m.ood.ratio}× limit</strong></span>
              {m.ood.color_saturation != null && <span>Colour: <strong>{m.ood.color_saturation}</strong></span>}
            </div>
          )}
        </div>
      ) : (
      <div className="open3d-metrics-box">
        <div className="box-title">Scan processing</div>
        <div className="stats-mini-row">
          <span>Points: <strong>{m.points_processed?.toLocaleString() ?? "—"}</strong></span>
          <span>Outliers removed: <strong>{m.outliers_removed ?? "—"}</strong></span>
          <span>Alignment: <strong>{m.alignment ?? "—"}</strong></span>
          <span>Coverage: <strong>{m.projection_coverage != null ? fmtPct(m.projection_coverage * 100, 0) : "—"}</strong></span>
          {m.icp_rmse != null && <span>ICP RMSE: <strong>{m.icp_rmse}</strong></span>}
          {m.icp_fitness != null && <span>ICP fitness: <strong>{m.icp_fitness}</strong></span>}
        </div>
        {sv.p95 != null && (
          <div className="stats-mini-row">
            <span>Surface variation p95: <strong>{sv.p95}</strong></span>
            <span>Above reference: <strong>{fmtPct(sv.fraction_above_reference * 100)}</strong></span>
          </div>
        )}
      </div>
      )}

      <div className="localization-note">
        <strong>Localization:</strong> {scan.localization?.available ? "Defect regions are provided by the model." : "not available. The classifier reports a result for the whole part, not where a defect is."}
      </div>
    </>
  );
}

// ---------- timings ----------
const TIMING_ROWS = [
  ["parse_ms", "Parse & validate"],
  ["decode_ms", "Decode image"],
  ["align_ms", "Alignment"],
  ["outlier_filter_ms", "Outlier filtering"],
  ["surface_analysis_ms", "Surface analysis"],
  ["projection_ms", "Depth/normal projection"],
  ["inference_ms", "Model inference"],
  ["geometry_save_ms", "Geometry save"],
  ["server_total_ms", "Server total"],
];

export function TimingPanel({ timings, client }) {
  const t = timings || {};
  if (!Object.keys(t).length) return null;
  return (
    <table className="timing-table">
      <thead>
        <tr><th>Stage</th><th>Time</th></tr>
      </thead>
      <tbody>
        {client?.uploadMs != null && <tr><td>Upload (browser → server)</td><td>{fmtMs(client.uploadMs)}</td></tr>}
        {TIMING_ROWS.filter(([k]) => t[k] != null || k === "inference_ms").map(([k, label]) => (
          <tr key={k} className={k === "server_total_ms" ? "timing-total" : ""}>
            <td>{label}</td>
            <td>{t[k] != null ? fmtMs(t[k]) : "not run"}</td>
          </tr>
        ))}
        {client?.roundTripMs != null && <tr className="timing-total"><td>Round trip (incl. upload & response)</td><td>{fmtMs(client.roundTripMs)}</td></tr>}
        {client?.geometryLoadMs != null && <tr><td>Geometry download + decode</td><td>{fmtMs(client.geometryLoadMs)}</td></tr>}
      </tbody>
    </table>
  );
}

// ---------- analytics ----------
export function AnalyticsView({ pareto, shifts, activity, performance }) {
  const maxAct = Math.max(1, ...(activity || []).map((a) => a.total));
  const perf = performance?.stages_ms || {};
  return (
    <div className="pareto-view">
      <h3>Defect categories (rejected scans)</h3>
      {!pareto?.length ? (
        <p className="text-muted">No rejected scans recorded yet.</p>
      ) : (
        <div className="pareto-bars">
          {pareto.map((it) => (
            <div key={it.defect_type} className="pareto-row">
              <div className="pareto-name"><strong>{it.defect_type}</strong><span>{it.count} ({it.percentage}%)</span></div>
              <div className="pareto-bar-track"><div className="pareto-bar-fill" style={{ width: `${Math.min(100, it.percentage)}%` }} /></div>
              <div className="pareto-cum">Cumulative: <strong>{it.cumulative_pct}%</strong></div>
            </div>
          ))}
        </div>
      )}

      <h4 className="section-gap">Recent activity (last 24 h, UTC hours)</h4>
      {activity?.some((a) => a.total > 0) ? (
        <div className="activity-chart">
          {activity.map((a) => (
            <div key={a.hour_start} className="activity-col" title={`${new Date(a.hour_start).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}: ${a.total} scans, ${a.failed} failed`}>
              <div className="activity-bar" style={{ height: `${(a.total / maxAct) * 100}%` }}>
                <div className="activity-bar-fail" style={{ height: a.total ? `${(a.failed / a.total) * 100}%` : 0 }} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-muted">No inspections in the last 24 hours.</p>
      )}

      {shifts?.by_shift?.length > 0 && (
        <>
          <h4 className="section-gap">By shift</h4>
          <div className="shift-grid">
            {shifts.by_shift.map((s) => (
              <div key={s.shift} className="shift-card">
                <div className="shift-name">{s.shift}</div>
                <div className="shift-yield">Defect rate: <strong>{fmtPct(s.defect_rate_pct)}</strong></div>
                <div className="shift-counts">Total {s.total} · Pass {s.passed} · Fail {s.failed} · Review {s.needs_review}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {Object.keys(perf).length > 0 && (
        <>
          <h4 className="section-gap">Measured processing time (last {performance.sample_size} scans)</h4>
          <table className="timing-table">
            <thead><tr><th>Stage</th><th>Mean</th><th>p95</th><th>Max</th></tr></thead>
            <tbody>
              {TIMING_ROWS.filter(([k]) => perf[k]).map(([k, label]) => (
                <tr key={k}><td>{label}</td><td>{fmtMs(perf[k].mean)}</td><td>{fmtMs(perf[k].p95)}</td><td>{fmtMs(perf[k].max)}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

// ---------- history ----------
export function HistoryTable({ items, total, skip, limit, filter, setFilter, search, setSearch, onPage, onSelect, currentId, loading, error }) {
  const FILTERS = [["all", "All"], ["PASS", "Pass"], ["FAIL", "Fail"], ["REVIEW", "Review"], ["UNCLASSIFIED", "Unclassified"]];
  return (
    <section className="stage-card history-section">
      <div className="card-header history-header">
        <div>
          <h2>INSPECTION LOG</h2>
          <span className="sub-text">{total} record{total === 1 ? "" : "s"} · stored in the database</span>
        </div>
        <div className="history-tools">
          <input className="select-input search-input" placeholder="Search part, file, batch, class…" value={search} onChange={(e) => setSearch(e.target.value)} />
          <div className="filter-pills">
            {FILTERS.map(([id, label]) => (
              <button key={id} className={`pill ${filter === id ? "active" : ""}`} onClick={() => setFilter(id)}>{label}</button>
            ))}
          </div>
        </div>
      </div>
      {error && <div className="banner banner-warning small">{error}</div>}
      <div className="table-responsive">
        <table className="scans-table">
          <thead>
            <tr><th>Part ID</th><th>File</th><th>Time</th><th>Status</th><th>Predicted class</th><th>Confidence</th><th>Processing</th><th>Batch</th><th /></tr>
          </thead>
          <tbody>
            {items.length === 0 ? (
              <tr><td colSpan="9" className="text-center text-muted">{loading ? "Loading…" : "No scans match."}</td></tr>
            ) : items.map((s) => (
              <tr key={s.id} className={`table-row ${s.id === currentId ? "selected-row" : ""}`} onClick={() => onSelect(s.id)}>
                <td><strong>{s.part_id}</strong></td>
                <td className="cell-ellipsis" title={s.filename}>{s.filename || "—"}</td>
                <td>{fmtTime(s.timestamp)}</td>
                <td><StatusPill status={s.status} /></td>
                <td>{s.defect_type || "—"}</td>
                <td>{fmtConf(s.confidence)}</td>
                <td>{fmtMs(s.latency_ms)}</td>
                <td>{s.batch_id}</td>
                <td><button className="btn-view" onClick={(e) => { e.stopPropagation(); onSelect(s.id); }}>Inspect</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <button className="btn btn-secondary" disabled={skip === 0} onClick={() => onPage(Math.max(0, skip - limit))}>← Newer</button>
        <span className="text-muted">{total === 0 ? "0" : `${skip + 1}–${Math.min(skip + limit, total)}`} of {total}</span>
        <button className="btn btn-secondary" disabled={skip + limit >= total} onClick={() => onPage(skip + limit)}>Older →</button>
      </div>
    </section>
  );
}
