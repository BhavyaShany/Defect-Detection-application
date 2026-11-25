import React, { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/**
 * 3D viewer for a processed scan.
 *
 * Props
 *  - geometry: { positions: Float32Array(3N), scores: Float32Array(N) in 0..1,
 *                outlier: Uint8Array(N), faces: Uint32Array(3F)|null,
 *                defectMask: Uint8Array(N)|null, bounds } | null
 *  - loading, error: provided by the parent
 *
 * Colour modes
 *  - "uniform": single colour
 *  - "surface": geometry-only surface variation (curvature / roughness / edges).
 *               This is NOT a defect map.
 *  - "defect" : only offered when the backend supplies a per-point defect mask. The current
 *               classifier has no localisation output, so this stays hidden -- nothing is faked.
 */

const STOPS = [
  [0.0, [0.12, 0.2, 0.55]],
  [0.25, [0.0, 0.7, 0.9]],
  [0.5, [0.1, 0.8, 0.3]],
  [0.75, [0.98, 0.85, 0.1]],
  [1.0, [0.95, 0.15, 0.1]],
];
const UNIFORM = [0.35, 0.65, 0.9];
const OUTLIER = [0.75, 0.3, 0.75];
const DIM = [0.35, 0.38, 0.45];
const DEFECT = [1.0, 0.1, 0.2];

function ramp(t, out) {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (x <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const k = (x - t0) / (t1 - t0);
      out[0] = c0[0] + (c1[0] - c0[0]) * k;
      out[1] = c0[1] + (c1[1] - c0[1]) * k;
      out[2] = c0[2] + (c1[2] - c0[2]) * k;
      return;
    }
  }
  out[0] = STOPS[4][1][0];
  out[1] = STOPS[4][1][1];
  out[2] = STOPS[4][1][2];
}

function disposeObject(obj) {
  if (!obj) return;
  obj.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      (Array.isArray(child.material) ? child.material : [child.material]).forEach((m) => m.dispose());
    }
  });
}

export default function ThreeViewer({ geometry, loading = false, error = null }) {
  const containerRef = useRef(null);
  const threeRef = useRef(null); // { scene, camera, renderer, controls, grid }
  const objectRef = useRef(null);
  const autoRotateRef = useRef(false);

  const [glError, setGlError] = useState(null);
  const [colorMode, setColorMode] = useState("surface");
  const [renderMode, setRenderMode] = useState("points"); // points | surface | wireframe
  const [pointSize, setPointSize] = useState(0.008);
  const [hideOutliers, setHideOutliers] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [shown, setShown] = useState(0);

  const hasFaces = !!(geometry && geometry.faces && geometry.faces.length > 0);
  const hasDefectMask = !!(geometry && geometry.defectMask);
  const valid = !!(geometry && geometry.positions && geometry.positions.length >= 9);

  // ---- one-time scene setup / teardown ----------------------------------
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch (e) {
      setGlError("WebGL is not available in this browser, so the 3D view cannot be shown.");
      return undefined;
    }
    const w = container.clientWidth || 600;
    const h = container.clientHeight || 450;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e17);
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
    camera.up.set(0, 0, 1); // scans are aligned with z as the depth axis
    camera.position.set(1.4, -1.8, 1.4);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.autoRotateSpeed = 1.2;

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(2, -3, 4);
    scene.add(key);

    const grid = new THREE.GridHelper(3, 30, 0x1e293b, 0x111827);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    threeRef.current = { scene, camera, renderer, controls, grid };

    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      controls.autoRotate = autoRotateRef.current;
      controls.update();
      renderer.render(scene, camera);
    };
    loop();

    const onResize = () => {
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      if (!cw || !ch) return;
      camera.aspect = cw / ch;
      camera.updateProjectionMatrix();
      renderer.setSize(cw, ch);
    };
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(onResize) : null;
    if (ro) ro.observe(container);
    else window.addEventListener("resize", onResize);

    const onContextLost = (e) => {
      e.preventDefault();
      setGlError("The WebGL context was lost. Reload the page to restore the 3D view.");
    };
    renderer.domElement.addEventListener("webglcontextlost", onContextLost);

    return () => {
      cancelAnimationFrame(frame);
      if (ro) ro.disconnect();
      else window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("webglcontextlost", onContextLost);
      controls.dispose();
      if (objectRef.current) {
        scene.remove(objectRef.current);
        disposeObject(objectRef.current);
        objectRef.current = null;
      }
      renderer.dispose();
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement);
      threeRef.current = null;
    };
  }, []);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  // Keep the selected modes valid for the current geometry
  useEffect(() => {
    if (!hasFaces && renderMode !== "points") setRenderMode("points");
    if (!hasDefectMask && colorMode === "defect") setColorMode("surface");
  }, [hasFaces, hasDefectMask, renderMode, colorMode]);

  // ---- camera fit -------------------------------------------------------
  const fitCamera = useCallback(() => {
    const t = threeRef.current;
    if (!t || !geometry || !geometry.bounds) return;
    const { min, max } = geometry.bounds;
    const center = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const radius = Math.max(0.2, 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]));
    const dist = radius / Math.sin((t.camera.fov * Math.PI) / 360);
    t.camera.position.copy(center).add(new THREE.Vector3(0.55, -0.75, 0.6).normalize().multiplyScalar(dist * 1.05));
    t.camera.near = Math.max(0.001, dist / 100);
    t.camera.far = dist * 20;
    t.camera.updateProjectionMatrix();
    t.controls.target.copy(center);
    t.controls.minDistance = radius * 0.2;
    t.controls.maxDistance = dist * 6;
    t.controls.update();
    t.grid.position.z = min[2] - 0.02;
  }, [geometry]);

  // Fit once per new geometry
  useEffect(() => {
    if (valid) fitCamera();
  }, [valid, fitCamera]);

  // ---- (re)build the displayed object ----------------------------------
  useEffect(() => {
    const t = threeRef.current;
    if (!t) return;
    if (objectRef.current) {
      t.scene.remove(objectRef.current);
      disposeObject(objectRef.current);
      objectRef.current = null;
    }
    setShown(0);
    if (!valid) return;

    const { positions, scores, outlier, faces, defectMask } = geometry;
    const n = positions.length / 3;
    const colors = new Float32Array(n * 3);
    const tmp = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      let c;
      if (outlier && outlier[i]) c = OUTLIER;
      else if (colorMode === "defect" && defectMask) c = defectMask[i] ? DEFECT : DIM;
      else if (colorMode === "surface" && scores) {
        ramp(scores[i], tmp);
        c = tmp;
      } else c = UNIFORM;
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
    }

    let obj;
    if (renderMode !== "points" && hasFaces) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      g.setIndex(new THREE.BufferAttribute(faces, 1));
      g.computeVertexNormals();
      obj = new THREE.Mesh(
        g,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          roughness: 0.55,
          metalness: 0.3,
          wireframe: renderMode === "wireframe",
        })
      );
      setShown(n);
    } else {
      let pos = positions;
      let col = colors;
      let count = n;
      if (hideOutliers && outlier) {
        const keep = [];
        for (let i = 0; i < n; i++) if (!outlier[i]) keep.push(i);
        pos = new Float32Array(keep.length * 3);
        col = new Float32Array(keep.length * 3);
        keep.forEach((src, j) => {
          pos.set(positions.subarray(src * 3, src * 3 + 3), j * 3);
          col.set(colors.subarray(src * 3, src * 3 + 3), j * 3);
        });
        count = keep.length;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
      obj = new THREE.Points(g, new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true }));
      setShown(count);
    }
    t.scene.add(obj);
    objectRef.current = obj;
    // pointSize is applied by a separate effect so dragging the slider does not rebuild buffers
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, valid, colorMode, renderMode, hideOutliers, hasFaces]);

  useEffect(() => {
    const obj = objectRef.current;
    if (obj && obj.isPoints) obj.material.size = pointSize;
  }, [pointSize, shown]);

  const btn = (active) => `viewer-btn${active ? " active" : ""}`;
  const overlay = glError || error || (loading ? "Loading 3D geometry…" : !valid ? "No scan loaded. Upload a scan or select one from the log." : null);

  return (
    <div className="viewer-root">
      <div ref={containerRef} className="viewer-canvas" />

      {overlay && <div className="viewer-overlay">{overlay}</div>}

      {valid && !glError && (
        <>
          <div className="viewer-toolbar">
            <button className={btn(autoRotate)} onClick={() => setAutoRotate((v) => !v)}>
              {autoRotate ? "⏸ Spin" : "▶ Spin"}
            </button>
            <button className="viewer-btn" onClick={fitCamera}>↺ Reset view</button>
            <span className="viewer-sep" />
            <button className={btn(renderMode === "points")} onClick={() => setRenderMode("points")}>Points</button>
            {hasFaces && (
              <>
                <button className={btn(renderMode === "surface")} onClick={() => setRenderMode("surface")}>Surface</button>
                <button className={btn(renderMode === "wireframe")} onClick={() => setRenderMode("wireframe")}>Wire</button>
              </>
            )}
            <span className="viewer-sep" />
            <button className={btn(colorMode === "uniform")} onClick={() => setColorMode("uniform")}>Original</button>
            <button className={btn(colorMode === "surface")} onClick={() => setColorMode("surface")}>Surface variation</button>
            {hasDefectMask && (
              <button className={btn(colorMode === "defect")} onClick={() => setColorMode("defect")}>Defect overlay</button>
            )}
            {renderMode === "points" && (
              <>
                <span className="viewer-sep" />
                <label className="viewer-slider" title="Point size">
                  Size
                  <input type="range" min="0.002" max="0.03" step="0.001" value={pointSize}
                    onChange={(e) => setPointSize(parseFloat(e.target.value))} />
                </label>
                <label className="viewer-check">
                  <input type="checkbox" checked={hideOutliers} onChange={(e) => setHideOutliers(e.target.checked)} />
                  Hide outliers
                </label>
              </>
            )}
          </div>

          <div className="viewer-info">{shown.toLocaleString()} points shown</div>

          {colorMode === "surface" && (
            <div className="viewer-legend">
              <div className="legend-title">Surface variation (geometry only)</div>
              <div className="legend-bar" />
              <div className="legend-labels"><span>flat</span><span>edges / pits / rough</span></div>
              <div className="legend-note">
                Not a defect map: it measures local curvature and cannot tell whether a feature is a defect.
              </div>
            </div>
          )}
          {colorMode === "defect" && (
            <div className="viewer-legend">
              <div className="legend-title">Model-reported defect region</div>
              <div className="legend-labels"><span style={{ color: "#ff1e40" }}>■ defect</span><span>■ other</span></div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
