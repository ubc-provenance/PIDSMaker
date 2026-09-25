/* scene.js — Three.js scene, temporal point shader, cameras, lines, picking */
const Scene = (() => {
  let renderer, scene, camera, controls, canvas;
  let points, material, geom;
  let highlight, hlGeom, hlMat;      // selected-node white layer
  let trajLine, trajGeom;            // temporal trajectory
  let attackLines, attackGeom;       // attack-graph edges
  let axis;
  let n = 0, H = 1;
  let curHop = 0, resetOnFirst = true, clock = 0;   // clock (s) drives the load-in fade
  let curPos = null;                 // Float32Array(cap*3) current hop (draw range = n)
  let decoded = null;                // accumulator: {pos[hop], attrs, ids, meta}
  let is2D = false;
  let center = new THREE.Vector3();
  let maxCoord = 10;
  let onPick = null;

  const VERT = `
    attribute vec4 aColor;
    attribute float aSize;
    attribute float aTwStart;
    attribute float aTwEnd;
    attribute float aVisible;
    attribute float aBorn;
    uniform float uTime, uSizeScale, uFlatten, uDpr, uDim, uClock;
    varying vec4 vColor;
    void main() {
      vec4 col = aColor;
      float size = aSize;
      vec3 p = position;
      p.z *= (1.0 - uFlatten);
      if (uTime >= 0.0) {
        float age = uTime - aTwStart;
        if (uTime < aTwStart || uTime >= aTwEnd) { col.a = 0.0; size = 0.0; }
        else {
          float blend = clamp(1.0 - (age / 1.5), 0.0, 1.0);
          col.rgb = mix(col.rgb, vec3(1.0, 1.0, 0.8), blend);
          col.a *= clamp(1.0 - (age * 0.3), 0.5, 1.0);
        }
      }
      if (uDim > 0.5) col.a = max(col.a * 0.55, 0.12);
      if (aVisible < 0.5) { size = 0.0; col.a = 0.0; }
      // load-in: freshly-streamed chunks grow + brighten over ~0.7s instead of
      // popping in. A brief white flash on arrival makes the fill feel alive.
      float born = clamp((uClock - aBorn) / 0.7, 0.0, 1.0);
      size *= smoothstep(0.0, 1.0, born);
      col.a *= born;
      col.rgb = mix(vec3(0.8, 0.95, 1.0), col.rgb, born);
      vColor = col;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      // fixed pixel size (parity with native VisPy markers) — crisp, no blow-up
      gl_PointSize = max(size * uSizeScale * uDpr, 0.0);
      gl_Position = projectionMatrix * mv;
    }`;

  const FRAG = `
    varying vec4 vColor;
    void main() {
      vec2 c = gl_PointCoord - vec2(0.5);
      float d = dot(c, c);
      if (d > 0.25) discard;
      // Solid disc with a thin antialias rim. depthWrite is ON (occlusion follows
      // true 3D depth), so the faint translucent rim must NOT write depth — a
      // low-coverage edge fragment would punch a dark "ghost hole" into whatever
      // point sits just behind it. Discard the rim; only the solid core writes
      // depth, which keeps clusters clean and bright instead of mottled/dark.
      float cov = smoothstep(0.25, 0.20, d);   // ~1 in the body, ~0 at the rim
      if (cov < 0.35) discard;
      gl_FragColor = vec4(vColor.rgb, vColor.a * cov);
    }`;

  // GPU-picking shaders (WebGL2 / GLSL3). Mirror VERT's transform exactly — flatten,
  // temporal-window cull, visibility cull, grow-in size — so a point lands on the
  // same pixel and at the same size as what's drawn, and hidden/culled points aren't
  // pickable. The fragment writes the vertex index (+1, so 0 = "no hit") packed into
  // RGBA, which we read back from one small block under the cursor. This makes a
  // click O(1) on the CPU instead of projecting all n points every time.
  const PICK_VERT = `
    precision highp float;
    precision highp int;
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    in vec3 position;
    in float aSize;
    in float aTwStart;
    in float aTwEnd;
    in float aVisible;
    in float aBorn;
    uniform float uTime, uSizeScale, uFlatten, uDpr, uClock;
    flat out vec4 vId;
    void main() {
      float size = aSize;
      vec3 p = position;
      p.z *= (1.0 - uFlatten);
      if (uTime >= 0.0 && (uTime < aTwStart || uTime >= aTwEnd)) size = 0.0;
      if (aVisible < 0.5) size = 0.0;
      float born = clamp((uClock - aBorn) / 0.7, 0.0, 1.0);
      size *= smoothstep(0.0, 1.0, born);
      int id = gl_VertexID + 1;                       // +1 so index 0 is distinguishable from the empty background
      vId = vec4(float(id & 255), float((id >> 8) & 255), float((id >> 16) & 255), float((id >> 24) & 255)) / 255.0;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = max(size * uSizeScale * uDpr, 0.0);
      gl_Position = size > 0.0 ? projectionMatrix * mv : vec4(2.0, 2.0, 2.0, 1.0);  // push culled points off-screen
    }`;
  const PICK_FRAG = `
    precision highp float;
    flat in vec4 vId;
    out vec4 pc;
    void main() {
      vec2 c = gl_PointCoord - vec2(0.5);
      if (dot(c, c) > 0.25) discard;                  // same disc as the visible marker
      pc = vId;
    }`;

  function init(cv) {
    canvas = cv;
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x050508, 1);
    scene = new THREE.Scene();
    makeCamera3D();
    window.addEventListener('resize', onResize);
    onResize();
    animate();
  }

  function aspect() {
    return canvas.clientWidth / Math.max(canvas.clientHeight, 1);
  }

  function makeCamera3D() {
    camera = new THREE.PerspectiveCamera(45, aspect(), 0.1, 100000);
    rebuildControls();
    // 3D: left-drag orbits, right-drag pans (OrbitControls defaults)
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    is2D = false;
  }

  function makeCamera2D() {
    const d = maxCoord * 1.2 || 10;
    const a = aspect();
    camera = new THREE.OrthographicCamera(-d * a, d * a, d, -d, -100000, 100000);
    rebuildControls();
    controls.enableRotate = false;
    controls.screenSpacePanning = true;
    controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    is2D = true;
  }

  function rebuildControls() {
    if (controls) controls.dispose();
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.target.copy(center);
  }

  function onResize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    if (material) material.uniforms.uDpr.value = renderer.getPixelRatio();
    if (camera.isPerspectiveCamera) {
      camera.aspect = aspect();
    } else {
      const d = maxCoord * 1.2 || 10, a = aspect();
      camera.left = -d * a; camera.right = d * a; camera.top = d; camera.bottom = -d;
    }
    camera.updateProjectionMatrix();
  }

  // Allocate the point cloud to `cap` nodes up front, rendering 0; chunks are then
  // streamed in with appendCore/appendAux so a big run fills in progressively.
  function beginLoad(cap, Hn, resetCam = true) {
    n = 0; H = Hn; resetOnFirst = resetCam; curHop = H - 1;
    const posHops = [];
    for (let h = 0; h < H; h++) posHops.push(new Float32Array(cap * 3));
    decoded = {
      n: cap, H, pos: posHops,
      attrs: new Float32Array(cap * 5),
      ids: new Uint32Array(cap),      // node_id
      meta: new Uint8Array(cap),      // packed label|det|type
    };
    curPos = decoded.pos[curHop];
    if (points) { scene.remove(points); geom.dispose(); material.dispose(); }

    geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(curPos, 3));
    geom.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(cap * 4), 4));
    geom.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(cap), 1));
    geom.setAttribute('aTwStart', new THREE.BufferAttribute(new Float32Array(cap), 1));
    geom.setAttribute('aTwEnd', new THREE.BufferAttribute(new Float32Array(cap), 1));
    geom.setAttribute('aVisible', new THREE.BufferAttribute(new Float32Array(cap), 1));
    geom.setAttribute('aBorn', new THREE.BufferAttribute(new Float32Array(cap), 1));
    geom.setDrawRange(0, 0);

    material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: -1.0 }, uSizeScale: { value: 1.4 }, uFlatten: { value: 0.0 },
        uDpr: { value: renderer.getPixelRatio() }, uDim: { value: 0.0 }, uClock: { value: clock },
      },
      vertexShader: VERT, fragmentShader: FRAG,
      // depthWrite:true → occlusion follows true 3D depth, not buffer draw order.
      // A point nearer the camera hides ones behind it regardless of the order
      // they were streamed in, so malicious (red) points in front of the cluster
      // stay visible instead of being painted over by benign points that are
      // actually further back. The fragment shader discards the transparent disc
      // edge (and not-yet-grown points), so only solid centres write depth.
      transparent: true, depthTest: true, depthWrite: true,
      blending: THREE.NormalBlending,
    });
    points = new THREE.Points(geom, material);
    points.frustumCulled = false;
    scene.add(points);
    setupAuxLayers();
  }

  // upload ONLY this chunk's slice of an attribute (bufferSubData), not the whole
  // cap-sized buffer — otherwise streaming N nodes in chunks is O(N²).
  function setRange(attr, comp, offset, count) {
    attr.updateRange = { offset: offset * comp, count: count * comp };
    attr.needsUpdate = true;
  }

  // TIER 1: positions + ids + meta. Grows the draw range and stamps the grow-in
  // clock; aColor/aSize/aVisible are written by the app's renderChunk (from the
  // meta byte), so this is everything needed to SHOW the cloud. aTwStart/aTwEnd
  // stay 0 until appendAux, which is fine — with the time uniform at "All" (-1)
  // every point is visible regardless.
  function appendCore(dec, offset) {
    for (let h = 0; h < H; h++) decoded.pos[h].set(dec.pos[h], offset * 3);
    decoded.meta.set(dec.meta, offset);
    geom.attributes.aBorn.array.fill(clock, offset, offset + dec.n);
    n = offset + dec.n;
    invalidateGrid();               // cloud grew — knn grid is stale
    geom.setDrawRange(0, n);
    setRange(geom.attributes.position, 3, offset, dec.n);
    setRange(geom.attributes.aBorn, 1, offset, dec.n);
    // Bounds/camera are framed from the first chunk only (a representative
    // sample); recomputing the full-cloud bbox every chunk would be O(N²).
    if (offset === 0) updateBounds(resetOnFirst);
  }

  // TIER 2: attrs. Fills the per-node time-window start/end so the time slider,
  // playback and temporal fade work; the rest of attrs (score) is read by the app
  // via attr(). Loaded in the background after the cloud is on screen.
  function appendAux(dec, offset) {
    decoded.attrs.set(dec.attrs, offset * 5);
    decoded.ids.set(dec.ids, offset);
    const ts = geom.attributes.aTwStart.array, te = geom.attributes.aTwEnd.array;
    for (let i = 0; i < dec.n; i++) {
      const o = offset + i;
      ts[o] = dec.attrs[i * 5 + 1]; te[o] = dec.attrs[i * 5 + 2];
    }
    setRange(geom.attributes.aTwStart, 1, offset, dec.n);
    setRange(geom.attributes.aTwEnd, 1, offset, dec.n);
  }

  function setupAuxLayers() {
    [highlight, trajLine, attackLines, axis].forEach((o) => { if (o) scene.remove(o); });
    // highlight (white) points
    hlGeom = new THREE.BufferGeometry();
    hlGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
    hlMat = new THREE.PointsMaterial({ color: 0xffffff, size: 14, sizeAttenuation: false, transparent: true, opacity: 0.85 });
    highlight = new THREE.Points(hlGeom, hlMat); highlight.frustumCulled = false; highlight.visible = false;
    scene.add(highlight);
    // trajectory
    trajGeom = new THREE.BufferGeometry();
    trajLine = new THREE.Line(trajGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true }));
    trajLine.frustumCulled = false; trajLine.visible = false; scene.add(trajLine);
    // attack edges
    attackGeom = new THREE.BufferGeometry();
    attackLines = new THREE.LineSegments(attackGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true }));
    attackLines.frustumCulled = false; attackLines.visible = false; scene.add(attackLines);
    // axis
    axis = new THREE.AxesHelper(1); scene.add(axis);
  }

  function updateBounds(resetCam) {
    // Robust core centering (parity with native update_spatial_bounds):
    // find the per-axis median, drop the farthest 0.5% of points from it,
    // then center + size on the bbox of the remaining cluster mass. This
    // keeps the axis on the clusters regardless of where a run's UMAP lands.
    const N = n;
    const step = Math.max(1, Math.floor(N / 60000)); // sample for percentiles
    const xs = [], ys = [], zs = [];
    for (let i = 0; i < N; i += step) {
      xs.push(curPos[i * 3]); ys.push(curPos[i * 3 + 1]); zs.push(curPos[i * 3 + 2]);
    }
    const median = (a) => { a.sort((p, q) => p - q); return a[Math.floor(a.length / 2)] || 0; };
    const mxd = median(xs), myd = median(ys), mzd = median(zs);
    const ds = [];
    for (let i = 0; i < N; i += step) {
      const dx = curPos[i * 3] - mxd, dy = curPos[i * 3 + 1] - myd, dz = curPos[i * 3 + 2] - mzd;
      ds.push(dx * dx + dy * dy + dz * dz);
    }
    ds.sort((p, q) => p - q);
    const thresh = ds[Math.floor(ds.length * 0.995)] ?? ds[ds.length - 1] ?? Infinity;
    // bbox of the core over the full point set
    let minx = Infinity, miny = Infinity, minz = Infinity, maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
    for (let i = 0; i < N; i++) {
      const x = curPos[i * 3], y = curPos[i * 3 + 1], z = curPos[i * 3 + 2];
      const dx = x - mxd, dy = y - myd, dz = z - mzd;
      if (dx * dx + dy * dy + dz * dz > thresh) continue;
      if (x < minx) minx = x; if (x > maxx) maxx = x;
      if (y < miny) miny = y; if (y > maxy) maxy = y;
      if (z < minz) minz = z; if (z > maxz) maxz = z;
    }
    if (!isFinite(minx)) { minx = maxx = mxd; miny = maxy = myd; minz = maxz = mzd; }
    center.set((minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2);
    maxCoord = Math.max((maxx - minx) / 2, (maxy - miny) / 2, (maxz - minz) / 2, 1);
    axis.scale.setScalar(Math.max(maxCoord / 3, 5));
    axis.position.copy(center);
    if (resetCam) resetCamera();
  }

  function resetCamera() {
    const d = maxCoord * 2.8 || 80;
    controls.target.copy(center);
    if (camera.isPerspectiveCamera) {
      // elevation ~30°, azimuth 0
      camera.position.set(center.x, center.y - d * 0.5, center.z + d * 0.87);
    } else {
      camera.position.set(center.x, center.y, center.z + 1000);
      onResize();
    }
    camera.lookAt(center);
    controls.update();
  }

  function set2D(flag) {
    if (flag === is2D) return;
    if (flag) { makeCamera2D(); material.uniforms.uFlatten.value = 1.0; }
    else { makeCamera3D(); material.uniforms.uFlatten.value = 0.0; }
    resetCamera();
  }

  function setHop(h) {
    h = Math.max(0, Math.min(H - 1, h));
    curHop = h;
    curPos = decoded.pos[h];
    invalidateGrid();               // positions changed with the hop — rebuild knn grid
    geom.setAttribute('position', new THREE.BufferAttribute(curPos, 3));
    geom.setDrawRange(0, n);
    geom.attributes.position.needsUpdate = true;
    updateBounds(false);
  }

  function setTime(t) { material.uniforms.uTime.value = t; }
  function setDim(on) { material.uniforms.uDim.value = on ? 1.0 : 0.0; }

  /* Write per-vertex display color (Float32 n*4), size (n), visible (n) — full. */
  function updateDisplay(color, size, visible) {
    if (color) { const a = geom.attributes.aColor; a.array.set(color); a.updateRange = { offset: 0, count: -1 }; a.needsUpdate = true; }
    if (size) { const a = geom.attributes.aSize; a.array.set(size); a.updateRange = { offset: 0, count: -1 }; a.needsUpdate = true; }
    if (visible) { const a = geom.attributes.aVisible; a.array.set(visible); a.updateRange = { offset: 0, count: -1 }; a.needsUpdate = true; }
  }

  /* Same, but only rows [offset, offset+count) — copies and uploads just that
     slice. Used per chunk during load so streaming is O(total), not O(n²). */
  function updateDisplayRange(offset, count, color, size, visible) {
    const put = (attr, comp, src) => {
      attr.array.set(src.subarray(offset * comp, (offset + count) * comp), offset * comp);
      attr.updateRange = { offset: offset * comp, count: count * comp };
      attr.needsUpdate = true;
    };
    put(geom.attributes.aColor, 4, color);
    put(geom.attributes.aSize, 1, size);
    put(geom.attributes.aVisible, 1, visible);
  }
  function refreshBounds() { updateBounds(false); }

  function setHighlight(positions) {
    if (!positions || positions.length === 0) { highlight.visible = false; return; }
    hlGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    hlGeom.attributes.position.needsUpdate = true;
    highlight.visible = true;
  }

  function setTrajectory(pos, col) {
    if (!pos || pos.length < 6) { trajLine.visible = false; return; }
    trajGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    trajGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(col), 4));
    trajLine.visible = true;
  }

  function setAttackEdges(pos, col) {
    if (!pos || pos.length < 6) { attackLines.visible = false; return; }
    attackGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    attackGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(col), 4));
    attackLines.visible = true;
  }

  function setCenterOffset(fx, fy, fz) {
    // pan sliders: center + frac*maxCoord per axis
    const c = new THREE.Vector3(
      center.x + fx * maxCoord, center.y + fy * maxCoord, center.z + fz * maxCoord);
    controls.target.copy(c); controls.update();
  }

  /* CPU pick: nearest visible point within 10px of (sx,sy) canvas coords. */
  function pick(sx, sy, visible) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const v = new THREE.Vector3();
    let best = -1, bestD = 196; // 14px squared
    const flat = material.uniforms.uFlatten.value;
    for (let i = 0; i < n; i++) {
      if (visible[i] < 0.5) continue;
      v.set(curPos[i * 3], curPos[i * 3 + 1], curPos[i * 3 + 2] * (1 - flat));
      v.project(camera);
      if (v.z > 1) continue; // behind
      const px = (v.x * 0.5 + 0.5) * w, py = (-v.y * 0.5 + 0.5) * h;
      const dx = px - sx, dy = py - sy, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function nodePos(i) {
    const flat = material.uniforms.uFlatten.value;
    return [curPos[i * 3], curPos[i * 3 + 1], curPos[i * 3 + 2] * (1 - flat)];
  }

  // ---- GPU picking (O(1) CPU) -----------------------------------------------
  let pickTarget = null, pickMat = null;
  function ensurePickResources() {
    if (!renderer.capabilities.isWebGL2) return false;    // GLSL3/gl_VertexID unavailable → caller falls back to CPU
    const w = canvas.width, h = canvas.height;            // device pixels (renderer sets these)
    if (!pickTarget || pickTarget.width !== w || pickTarget.height !== h) {
      if (pickTarget) pickTarget.dispose();
      pickTarget = new THREE.WebGLRenderTarget(w, h, {
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat, type: THREE.UnsignedByteType, depthBuffer: true,
      });
    }
    if (!pickMat) {
      // RawShaderMaterial: Three injects no boilerplate, so the GLSL3 shader has a
      // single, unambiguous fragment out. The renderer still binds `position` and
      // the built-in matrices by name.
      pickMat = new THREE.RawShaderMaterial({
        uniforms: { uTime: { value: -1 }, uSizeScale: { value: 1.4 }, uFlatten: { value: 0 }, uDpr: { value: 1 }, uClock: { value: 0 } },
        vertexShader: PICK_VERT, fragmentShader: PICK_FRAG, glslVersion: THREE.GLSL3,
        depthTest: true, depthWrite: true, transparent: false,
      });
    }
    return true;
  }

  /* Nearest point to (sx,sy) via an offscreen index render. Returns the buffer
     row, -1 for empty space, or -2 if GPU picking isn't available (WebGL1) so the
     caller uses the CPU pick(). */
  function pickGPU(sx, sy) {
    if (!points || n === 0) return -1;
    if (!ensurePickResources()) return -2;
    const savedVis = [];
    const prevMat = points.material;
    const savedColor = renderer.getClearColor(new THREE.Color()), savedAlpha = renderer.getClearAlpha();
    try {
      // mirror the live material's uniforms so culling/positions match the screen
      const u = material.uniforms;
      pickMat.uniforms.uTime.value = u.uTime.value;
      pickMat.uniforms.uSizeScale.value = u.uSizeScale.value;
      pickMat.uniforms.uFlatten.value = u.uFlatten.value;
      pickMat.uniforms.uDpr.value = u.uDpr.value;
      pickMat.uniforms.uClock.value = u.uClock.value;
      // render ONLY the points (other layers' materials don't encode ids) into the target
      for (const ch of scene.children) { savedVis.push([ch, ch.visible]); if (ch !== points) ch.visible = false; }
      points.material = pickMat;
      renderer.setRenderTarget(pickTarget);
      renderer.setClearColor(0x000000, 0.0); renderer.clear();
      renderer.render(scene, camera);
      // read a small block around the cursor and take the nearest non-empty pixel
      const pr = renderer.getPixelRatio();
      const W = pickTarget.width, H = pickTarget.height;
      const cxp = Math.round(sx * pr), cyp = Math.round(sy * pr);        // device px, top-left origin
      const R = Math.min(20, Math.max(2, Math.round(10 * pr)));          // ~10 CSS-px click tolerance
      const x0 = Math.max(0, cxp - R), x1 = Math.min(W, cxp + R + 1);
      const y0 = Math.max(0, cyp - R), y1 = Math.min(H, cyp + R + 1);
      let found = -1;
      const bw = x1 - x0, bh = y1 - y0;
      if (bw > 0 && bh > 0) {
        const gy0 = H - y1;                                              // GL origin is bottom-left
        const buf = new Uint8Array(bw * bh * 4);
        renderer.readRenderTargetPixels(pickTarget, x0, gy0, bw, bh, buf);
        let bestD = Infinity;
        for (let yy = 0; yy < bh; yy++) {
          for (let xx = 0; xx < bw; xx++) {
            const o = (yy * bw + xx) * 4;
            const id = buf[o] + buf[o + 1] * 256 + buf[o + 2] * 65536 + buf[o + 3] * 16777216;
            if (id === 0) continue;                                     // background
            const devX = x0 + xx, devY = H - 1 - (gy0 + yy);            // back to top-left device px
            const dd = (devX - cxp) * (devX - cxp) + (devY - cyp) * (devY - cyp);
            if (dd < bestD) { bestD = dd; found = id - 1; }
          }
        }
      }
      return found;
    } catch (e) {
      return -2;   // anything unexpected on the GPU path → caller uses the CPU pick
    } finally {
      renderer.setRenderTarget(null);
      renderer.setClearColor(savedColor, savedAlpha);
      points.material = prevMat;
      for (const [ch, v] of savedVis) ch.visible = v;
    }
  }

  /* Upload ONLY the aVisible attribute — used on selection, where colours/sizes
     don't change, so re-copying+uploading the full aColor (4·n floats) every
     click is pure waste. */
  function updateVisible(visible) {
    const a = geom.attributes.aVisible;
    a.array.set(visible);
    a.updateRange = { offset: 0, count: -1 };
    a.needsUpdate = true;
  }

  /* Sync only `indices` of aVisible from `visible`, then upload the single
     bounding sub-range that covers them. On selection only a handful of rows
     change, so this avoids the full n-length copy + upload updateVisible() does.
     (A node's time-window instances are usually close together in buffer order,
     so the range stays small; worst case it is still ≤ a full upload.) */
  function updateVisibleAt(visible, indices) {
    if (!indices.length) return;
    const a = geom.attributes.aVisible, arr = a.array;
    let lo = Infinity, hi = -1;
    for (const i of indices) { arr[i] = visible[i]; if (i < lo) lo = i; if (i > hi) hi = i; }
    a.updateRange = { offset: lo, count: hi - lo + 1 };
    a.needsUpdate = true;
  }

  // ---- spatial grid for k-nearest-neighbour queries -------------------------
  // Selection used to scan all n points to find the 20 nearest — O(n) per click,
  // a multi-hundred-ms freeze at millions of nodes. Instead bucket points into a
  // uniform grid once (counting-sort into typed arrays, O(n) build, ~O(n) memory)
  // and answer a query by expanding shells of cells outward from the target until
  // the k nearest are provably found. Rebuilt lazily when the cloud grows or the
  // hop (=positions) changes.
  let grid = null, gridDirty = true;
  function invalidateGrid() { gridDirty = true; }

  function buildGrid() {
    grid = null;
    if (n === 0 || !curPos) return;
    let minx = Infinity, miny = Infinity, minz = Infinity;
    let maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = curPos[i * 3], y = curPos[i * 3 + 1], z = curPos[i * 3 + 2];
      if (x < minx) minx = x; if (x > maxx) maxx = x;
      if (y < miny) miny = y; if (y > maxy) maxy = y;
      if (z < minz) minz = z; if (z > maxz) maxz = z;
    }
    const ex = Math.max(maxx - minx, 1e-6), ey = Math.max(maxy - miny, 1e-6), ez = Math.max(maxz - minz, 1e-6);
    // aim for ~2 points/cell; then clamp the cell count so the index stays small.
    let cell = Math.cbrt((ex * ey * ez) / Math.max(1, n / 2));
    if (!(cell > 0)) cell = Math.max(ex, ey, ez) / 32 || 1;
    let gx = Math.max(1, Math.ceil(ex / cell));
    let gy = Math.max(1, Math.ceil(ey / cell));
    let gz = Math.max(1, Math.ceil(ez / cell));
    while (gx * gy * gz > 4000000) { cell *= 1.26; gx = Math.max(1, Math.ceil(ex / cell)); gy = Math.max(1, Math.ceil(ey / cell)); gz = Math.max(1, Math.ceil(ez / cell)); }
    const cells = gx * gy * gz;
    const cellOf = (i) => {
      const ix = Math.min(gx - 1, ((curPos[i * 3] - minx) / cell) | 0);
      const iy = Math.min(gy - 1, ((curPos[i * 3 + 1] - miny) / cell) | 0);
      const iz = Math.min(gz - 1, ((curPos[i * 3 + 2] - minz) / cell) | 0);
      return ix + gx * (iy + gy * iz);
    };
    const starts = new Int32Array(cells + 1);
    for (let i = 0; i < n; i++) starts[cellOf(i) + 1]++;
    for (let c = 0; c < cells; c++) starts[c + 1] += starts[c];
    const order = new Int32Array(n);
    const cur = starts.slice(0, cells);
    for (let i = 0; i < n; i++) { const c = cellOf(i); order[cur[c]++] = i; }
    grid = { gx, gy, gz, cell, minx, miny, minz, starts, order };
  }

  /* Indices of the k nearest points to point `idx` (excluding itself). */
  function knnIndices(idx, K) {
    if (gridDirty) { buildGrid(); gridDirty = false; }
    const x = curPos[idx * 3], y = curPos[idx * 3 + 1], z = curPos[idx * 3 + 2];
    if (!grid || n <= K + 1) {
      // brute force for tiny clouds (or if the grid failed to build)
      const best = [];
      for (let i = 0; i < n; i++) {
        if (i === idx) continue;
        const dx = curPos[i * 3] - x, dy = curPos[i * 3 + 1] - y, dz = curPos[i * 3 + 2] - z;
        best.push([dx * dx + dy * dy + dz * dz, i]);
      }
      best.sort((a, b) => a[0] - b[0]);
      return best.slice(0, K).map((e) => e[1]);
    }
    const { gx, gy, gz, cell, minx, miny, minz, starts, order } = grid;
    const cx = Math.min(gx - 1, ((x - minx) / cell) | 0);
    const cy = Math.min(gy - 1, ((y - miny) / cell) | 0);
    const cz = Math.min(gz - 1, ((z - minz) / cell) | 0);
    // best[] kept sorted ascending by distance, length ≤ K
    const best = [];
    const consider = (j) => {
      if (j === idx) return;
      const dx = curPos[j * 3] - x, dy = curPos[j * 3 + 1] - y, dz = curPos[j * 3 + 2] - z;
      const d = dx * dx + dy * dy + dz * dz;
      if (best.length < K) {
        let p = best.length; while (p > 0 && best[p - 1][0] > d) p--;
        best.splice(p, 0, [d, j]);
      } else if (d < best[K - 1][0]) {
        let p = K - 1; while (p > 0 && best[p - 1][0] > d) p--;
        best.splice(K - 1, 1); best.splice(p, 0, [d, j]);
      }
    };
    const maxR = Math.max(gx, gy, gz);
    for (let r = 0; r <= maxR; r++) {
      for (let dz = -r; dz <= r; dz++) {
        const zz = cz + dz; if (zz < 0 || zz >= gz) continue;
        for (let dy = -r; dy <= r; dy++) {
          const yy = cy + dy; if (yy < 0 || yy >= gy) continue;
          for (let dx = -r; dx <= r; dx++) {
            // only the shell at Chebyshev radius r (interior was done at smaller r)
            if (Math.abs(dx) !== r && Math.abs(dy) !== r && Math.abs(dz) !== r) continue;
            const xx = cx + dx; if (xx < 0 || xx >= gx) continue;
            const c = xx + gx * (yy + gy * zz);
            for (let p = starts[c]; p < starts[c + 1]; p++) consider(order[p]);
          }
        }
      }
      // after finishing shell r, any unexamined point is ≥ r*cell away; stop once
      // we hold K and the Kth is nearer than that frontier.
      if (best.length >= K) {
        const frontier = r * cell;
        if (best[best.length - 1][0] <= frontier * frontier) break;
      }
    }
    return best.map((e) => e[1]);
  }

  // Render every frame. Gating renders on OrbitControls' change flag made
  // damped wheel-zoom stutter, so keep it simple and smooth.
  function animate() {
    requestAnimationFrame(animate);
    clock = performance.now() / 1000;
    if (material) material.uniforms.uClock.value = clock;
    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  return {
    init, beginLoad, appendCore, appendAux, setHop, setTime, setDim, set2D, updateDisplay,
    updateDisplayRange, updateVisible, updateVisibleAt, knnIndices, refreshBounds, setHighlight,
    setTrajectory, setAttackEdges, resetCamera, setCenterOffset, pick, pickGPU, nodePos,
    get curPos() { return curPos; },
    get decoded() { return decoded; },
  };
})();
