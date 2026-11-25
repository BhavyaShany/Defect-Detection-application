// Backend access layer: configurable base URL, consistent errors, upload with progress.

// Development (npm run dev): the API is at http://localhost:8000 unless VITE_API_BASE_URL says otherwise.
// Production build (served by the backend): same origin, so no URL needs to be configured.
const envBase = import.meta.env.VITE_API_BASE_URL;
export const API_BASE = (envBase !== undefined ? envBase : import.meta.env.DEV ? "http://localhost:8000" : "").replace(/\/$/, "");
export const WS_URL = API_BASE
  ? API_BASE.replace(/^http/, "ws") + "/ws/live"
  : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/live`;

export class ApiError extends Error {
  constructor(message, code = "ERROR", status = 0) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

/** Normalises the backend's {detail:{code,message}} / {detail:"..."} / FastAPI 422 list shapes. */
export function parseErrorBody(status, body) {
  const d = body && body.detail;
  if (d && typeof d === "object" && !Array.isArray(d) && d.message) return new ApiError(d.message, d.code || "ERROR", status);
  if (typeof d === "string") return new ApiError(d, "ERROR", status);
  if (Array.isArray(d)) return new ApiError(d.map((e) => e.msg).join("; "), "VALIDATION_ERROR", status);
  return new ApiError(`Request failed (HTTP ${status}).`, "ERROR", status);
}

export async function getJson(path, { signal, timeoutMs = 30000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  if (signal) signal.addEventListener("abort", () => ctrl.abort());
  try {
    const res = await fetch(`${API_BASE}${path}`, { signal: ctrl.signal });
    let body = null;
    try {
      body = await res.json();
    } catch (e) {
      /* non-JSON body */
    }
    if (!res.ok) throw parseErrorBody(res.status, body);
    return body;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (e.name === "AbortError") throw new ApiError("The request timed out or was cancelled.", "TIMEOUT");
    throw new ApiError("Cannot reach the inspection server. Is the backend running?", "NETWORK_ERROR");
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Upload a scan with progress. Resolves { data, uploadMs, roundTripMs }.
 *  - uploadMs: time until the browser finished sending the file
 *  - roundTripMs: total time until the final response arrived
 */
export function uploadScan(file, fields, { onUploadProgress, onUploadDone } = {}) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    Object.entries(fields || {}).forEach(([k, v]) => v && form.append(k, v));
    const xhr = new XMLHttpRequest();
    const t0 = performance.now();
    let uploadMs = null;
    xhr.open("POST", `${API_BASE}/api/scans/upload`);
    xhr.timeout = 10 * 60 * 1000;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onUploadProgress) onUploadProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.upload.onload = () => {
      uploadMs = performance.now() - t0;
      if (onUploadDone) onUploadDone();
    };
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch (e) {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve({ data: body, uploadMs, roundTripMs: performance.now() - t0 });
      } else {
        reject(parseErrorBody(xhr.status, body));
      }
    };
    xhr.onerror = () => reject(new ApiError("Cannot reach the inspection server. Is the backend running?", "NETWORK_ERROR"));
    xhr.ontimeout = () => reject(new ApiError("The upload timed out.", "TIMEOUT"));
    xhr.onabort = () => reject(new ApiError("The upload was cancelled.", "ABORTED"));
    xhr.send(form);
  });
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** Decodes the /geometry payload into typed arrays for ThreeViewer. */
export function decodeGeometry(p) {
  const pos = b64ToBytes(p.positions_b64);
  const sc = b64ToBytes(p.scores_b64);
  const out = {
    positions: new Float32Array(pos.buffer, 0, pos.byteLength / 4),
    scores: new Float32Array(sc.buffer, 0, sc.byteLength / 4),
    outlier: b64ToBytes(p.outlier_b64),
    faces: null,
    defectMask: null,
    bounds: p.bounds,
    pointCount: p.point_count,
  };
  if (p.faces_b64) {
    const f = b64ToBytes(p.faces_b64);
    out.faces = new Uint32Array(f.buffer, 0, f.byteLength / 4);
  }
  if (p.defect_mask_b64) out.defectMask = b64ToBytes(p.defect_mask_b64);
  return out;
}
