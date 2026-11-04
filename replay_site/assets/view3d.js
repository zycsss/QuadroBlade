/**
 * The 3D view: the same run seen in perspective, with an orbit camera.
 *
 * There is no WebGL and no third-party renderer here. The scene is line art, so a small painter's
 * algorithm - project every primitive, sort by distance, draw far to near - is enough, and it keeps
 * the whole site to files that work from `file://` and from any static server.
 *
 * What it adds over the top view is height: the gate openings are squares in space, the poles are
 * vertical capsules the drone has to pass beside, and the drone leaves a vertical drop line so its
 * altitude is readable at a glance.
 *
 * Exposed as `Replay.view3d.draw(canvas, model, state)` and `Replay.view3d.attachControls(canvas, ...)`.
 */
(function (root) {
  "use strict";

  const NS = (root.Replay = root.Replay || {});
  const {
    GEOMETRY,
    speedColor,
    rotate,
    prepareCanvas,
    label,
    add,
    sub,
    scale,
    cross,
    dot,
    length,
    normalize,
  } = NS.util;

  /** Vertical field of view in radians. */
  const FOV = 0.9;
  /** Distance below which a point is behind the camera and must not be drawn. */
  const NEAR = 0.08;

  const COLORS = {
    grid: "rgba(130, 160, 200, 0.055)",
    gridStrong: "rgba(130, 160, 200, 0.12)",
    bounds: "rgba(120, 160, 220, 0.25)",
    future: "rgba(190, 205, 225, 0.16)",
    gate: "#f0ad4e",
    gateDim: "rgba(240, 173, 78, 0.42)",
    gateOpening: "rgba(210, 215, 225, 0.30)",
    pole: "#b388ff",
    poleDim: "rgba(179, 136, 255, 0.40)",
    contact: "rgba(255, 105, 105, 0.45)",
    sensor: "rgba(80, 200, 255, 0.40)",
    nominalGhost: "rgba(160, 175, 195, 0.35)",
    drone: "#ffffff",
  };

  /**
   * Camera position and screen basis for one frame.
   *
   * @param {{azimuth: number, elevation: number, distance: number, target: number[]}} camera Orbit
   *     camera parameters.
   * @param {number} width Canvas width in CSS pixels.
   * @param {number} height Canvas height in CSS pixels.
   * @returns {Object} Basis with `eye`, `right`, `up`, `forward`, `focal` and the canvas centre.
   */
  function cameraBasis(camera, width, height) {
    const elevation = Math.min(1.4, Math.max(-1.4, camera.elevation));
    const cosElevation = Math.cos(elevation);
    const eye = [
      camera.target[0] + camera.distance * cosElevation * Math.cos(camera.azimuth),
      camera.target[1] + camera.distance * cosElevation * Math.sin(camera.azimuth),
      camera.target[2] + camera.distance * Math.sin(elevation),
    ];
    const forward = normalize(sub(camera.target, eye));
    let right = normalize(cross(forward, [0, 0, 1]));
    if (!Number.isFinite(right[0]) || length(right) < 1e-6) right = [0, 1, 0];
    const up = cross(right, forward);
    const focal = height * 0.5 / Math.tan(FOV / 2);
    return { eye, forward, right, up, focal, width, height };
  }

  /**
   * Project a world point to the canvas.
   *
   * @param {Object} basis Camera basis.
   * @param {number[]} point World position.
   * @returns {{x: number, y: number, depth: number}|null} Screen position and camera distance, or
   *     null when the point is behind the camera.
   */
  function project(basis, point) {
    const relative = sub(point, basis.eye);
    const depth = dot(relative, basis.forward);
    if (depth < NEAR) return null;
    const x = dot(relative, basis.right);
    const y = dot(relative, basis.up);
    const k = basis.focal / depth;
    return { x: basis.width / 2 + x * k, y: basis.height / 2 - y * k, depth };
  }

  /**
   * Default orbit camera for a model: far enough away for the whole track to fit.
   *
   * @param {Object} model Replay model.
   * @returns {{azimuth: number, elevation: number, distance: number, target: number[]}} Camera.
   */
  function defaultCamera(model) {
    const view = model.view;
    const dx = view.xmax - view.xmin;
    const dy = view.ymax - view.ymin;
    return {
      azimuth: -1.05,
      elevation: 0.42,
      // Close enough that the track fills the panel; the factor is a margin, not the fitting maths.
      distance: Math.max(2.4, (Math.max(dx, dy) * 1.05) / (2 * Math.tan(FOV / 2))),
      target: [(view.xmin + view.xmax) / 2, (view.ymin + view.ymax) / 2, 0.55],
    };
  }

  /**
   * Whether an object's true position is known at the step being shown.
   *
   * @param {{reveal: number}} object Gate or pole model.
   * @param {number} step Step being shown.
   * @param {boolean} revealedAll Whether the user asked to lift the reveal rule.
   * @returns {boolean} True when the true position may be drawn.
   */
  function isRevealed(object, step, revealedAll) {
    return revealedAll || (object.reveal >= 0 && step >= object.reveal);
  }

  /**
   * Draw one frame.
   *
   * @param {HTMLCanvasElement} canvas Canvas to draw on.
   * @param {Object} model Replay model.
   * @param {{step: number, revealedAll: boolean, camera: Object}} state What to show and from where.
   */
  function draw(canvas, model, state) {
    const { ctx, width, height } = prepareCanvas(canvas);
    const background = ctx.createLinearGradient(0, 0, 0, height);
    background.addColorStop(0, "#0d131b");
    background.addColorStop(1, "#131c26");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    const basis = cameraBasis(state.camera, width, height);
    // Every primitive carries the camera distance of its middle, which is all the painter's
    // algorithm needs to decide what covers what.
    const primitives = [];
    const overlays = [];

    const pushLine = (a, b, color, lineWidth, dash) => {
      const pa = project(basis, a);
      const pb = project(basis, b);
      if (!pa || !pb) return;
      primitives.push({ depth: (pa.depth + pb.depth) / 2, kind: "line", pa, pb, color, lineWidth, dash });
    };
    const pushPoly = (points, color, lineWidth, dash, close) => {
      const projected = points.map((point) => project(basis, point));
      if (projected.some((point) => !point)) return;
      const depth = projected.reduce((sum, point) => sum + point.depth, 0) / projected.length;
      primitives.push({ depth, kind: "poly", points: projected, color, lineWidth, dash, close: close !== false });
    };
    const pushPoint = (point, radius, color) => {
      const projected = project(basis, point);
      if (!projected) return;
      primitives.push({ depth: projected.depth, kind: "point", ...projected, radius, color });
    };
    const pushText = (point, text, color, font) => {
      const projected = project(basis, point);
      if (!projected) return;
      overlays.push({ depth: projected.depth, x: projected.x, y: projected.y, text, color, font });
    };

    drawFloor(model, pushLine, pushText);
    drawGates(model, state, pushLine, pushPoly, pushText, pushPoint);
    drawPoles(model, state, pushLine, pushPoly, pushText, pushPoint);

    const step = state.step;
    // The rest of the trajectory, faint, so the scene is readable before the replay reaches the end.
    if (step < model.n - 1) {
      pushPoly(model.pos.slice(step), COLORS.future, 1, null, false);
    }
    for (let i = 1; i <= step; i++) {
      pushLine(
        model.pos[i - 1],
        model.pos[i],
        speedColor(model.speedFraction(model.speed[i])),
        2.4,
        null
      );
    }

    drawSensorRange(model, step, pushPoly);
    drawDrone(model, step, pushLine, pushPoint, speedColor);

    primitives.sort((a, b) => b.depth - a.depth);
    for (const primitive of primitives) {
      if (primitive.kind === "line") {
        ctx.strokeStyle = primitive.color;
        ctx.lineWidth = primitive.lineWidth;
        ctx.setLineDash(primitive.dash || []);
        ctx.beginPath();
        ctx.moveTo(primitive.pa.x, primitive.pa.y);
        ctx.lineTo(primitive.pb.x, primitive.pb.y);
        ctx.stroke();
      } else if (primitive.kind === "poly") {
        ctx.strokeStyle = primitive.color;
        ctx.lineWidth = primitive.lineWidth;
        ctx.setLineDash(primitive.dash || []);
        ctx.beginPath();
        primitive.points.forEach((point, index) => {
          if (index === 0) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
        });
        if (primitive.close) ctx.closePath();
        ctx.stroke();
      } else if (primitive.kind === "point") {
        ctx.fillStyle = primitive.color;
        ctx.beginPath();
        ctx.arc(primitive.x, primitive.y, primitive.radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.setLineDash([]);
    for (const overlay of overlays) {
      label(ctx, overlay.text, overlay.x, overlay.y, {
        color: overlay.color,
        font: overlay.font || "10px ui-monospace, monospace",
      });
    }

    label(ctx, "drag to orbit · wheel to zoom · double-click to reset", 12, height - 14, {
      color: "rgba(150, 170, 195, 0.65)",
      font: "10px ui-monospace, monospace",
    });
  }

  /**
   * The floor grid and the metre labels along two edges.
   *
   * @param {Object} model Replay model.
   * @param {Function} pushLine Line primitive helper.
   * @param {Function} pushText Text overlay helper.
   */
  function drawFloor(model, pushLine, pushText) {
    const view = model.view;
    const z = 0;
    for (let x = Math.ceil(view.xmin * 2) / 2; x <= view.xmax; x += 0.5) {
      const major = Math.abs(x - Math.round(x)) < 1e-6;
      pushLine([x, view.ymin, z], [x, view.ymax, z], major ? COLORS.gridStrong : COLORS.grid, 1, null);
    }
    for (let y = Math.ceil(view.ymin * 2) / 2; y <= view.ymax; y += 0.5) {
      const major = Math.abs(y - Math.round(y)) < 1e-6;
      pushLine([view.xmin, y, z], [view.xmax, y, z], major ? COLORS.gridStrong : COLORS.grid, 1, null);
    }
    // Track boundary, so the drone's excursions have a frame of reference.
    pushLine([view.xmin, view.ymin, z], [view.xmax, view.ymin, z], COLORS.bounds, 1.2, null);
    pushLine([view.xmax, view.ymin, z], [view.xmax, view.ymax, z], COLORS.bounds, 1.2, null);
    pushLine([view.xmax, view.ymax, z], [view.xmin, view.ymax, z], COLORS.bounds, 1.2, null);
    pushLine([view.xmin, view.ymax, z], [view.xmin, view.ymin, z], COLORS.bounds, 1.2, null);
    pushText([view.xmax + 0.1, view.ymin, z], "x", "rgba(150, 170, 195, 0.7)", "11px ui-monospace, monospace");
    pushText([view.xmin, view.ymax + 0.1, z], "y", "rgba(150, 170, 195, 0.7)", "11px ui-monospace, monospace");
  }

  /**
   * Every gate: the square frame, the opening, the crossing direction and both positions.
   *
   * @param {Object} model Replay model.
   * @param {Object} state View state.
   * @param {Function} pushLine Line primitive helper.
   * @param {Function} pushPoly Polyline primitive helper.
   * @param {Function} pushText Text overlay helper.
   * @param {Function} pushPoint Point primitive helper.
   */
  function drawGates(model, state, pushLine, pushPoly, pushText, pushPoint) {
    const half = GEOMETRY.gateFrameHalf;
    const opening = GEOMETRY.gateOpeningHalf;
    const square = (centre, quat, size) => {
      const lateral = rotate(quat, [0, 1, 0]);
      const vertical = rotate(quat, [0, 0, 1]);
      const corner = (a, b) => add(centre, add(scale(lateral, a), scale(vertical, b)));
      return [
        corner(-size, -size),
        corner(size, -size),
        corner(size, size),
        corner(-size, size),
      ];
    };

    for (const gate of model.gates) {
      const revealed = isRevealed(gate, state.step, state.revealedAll);
      const centre = revealed ? gate.truth : gate.orig;
      const quat = revealed ? gate.truthQuat : gate.quat;
      const color = revealed ? COLORS.gate : COLORS.gateDim;
      const normal = rotate(quat, [1, 0, 0]);
      const lateral = rotate(quat, [0, 1, 0]);
      const vertical = rotate(quat, [0, 0, 1]);

      pushPoly(square(centre, quat, half), color, revealed ? 2.4 : 1.2, revealed ? null : [4, 4]);
      pushPoly(square(centre, quat, opening), COLORS.gateOpening, 1.1, [3, 3]);
      pushLine(centre, add(centre, scale(normal, 0.55)), color, 1.3, null);
      const tip = add(centre, scale(normal, 0.55));
      const head = 0.09;
      pushLine(tip, add(add(tip, scale(normal, -head)), scale(lateral, head * 0.7)), color, 1.3, null);
      pushLine(tip, add(add(tip, scale(normal, -head)), scale(lateral, -head * 0.7)), color, 1.3, null);
      pushText(
        add(centre, scale(vertical, half + 0.10)),
        `g${gate.id}${revealed ? "" : " ?"}`,
        color,
        "bold 11px ui-monospace, monospace"
      );

      if (revealed && gate.moved > 0.02) {
        // The nominal gate the controller planned on, and how far it really was from there.
        pushPoly(square(gate.orig, gate.quat, half), COLORS.nominalGhost, 1, [3, 4]);
        pushLine(gate.orig, gate.truth, COLORS.nominalGhost, 1, [2, 4]);
        pushPoint(gate.orig, 2, COLORS.nominalGhost);
      }
    }
  }

  /**
   * Every pole: a vertical capsule with the radius at which the drone would touch it.
   *
   * @param {Object} model Replay model.
   * @param {Object} state View state.
   * @param {Function} pushLine Line primitive helper.
   * @param {Function} pushPoly Polyline primitive helper.
   * @param {Function} pushText Text overlay helper.
   * @param {Function} pushPoint Point primitive helper.
   */
  function drawPoles(model, state, pushLine, pushPoly, pushText, pushPoint) {
    const contact = GEOMETRY.poleRadius + GEOMETRY.droneRadius;
    for (const pole of model.poles) {
      const revealed = isRevealed(pole, state.step, state.revealedAll);
      const position = revealed ? pole.truth : pole.orig;
      const top = revealed ? pole.top : Math.max(pole.orig[2], 0);
      const color = revealed ? COLORS.pole : COLORS.poleDim;
      pushLine([position[0], position[1], 0], [position[0], position[1], top], color, revealed ? 3 : 1.6, revealed ? null : [4, 4]);
      pushPoly(
        Array.from({ length: 28 }, (_, i) => {
          const angle = (i / 28) * Math.PI * 2;
          return [
            position[0] + contact * Math.cos(angle),
            position[1] + contact * Math.sin(angle),
            0.01,
          ];
        }),
        COLORS.contact,
        1,
        [3, 3]
      );
      pushPoint([position[0], position[1], top], revealed ? 2.4 : 1.6, color);
      pushText(
        [position[0], position[1], top + 0.09],
        `o${pole.id}${revealed ? "" : " ?"}`,
        color,
        "10px ui-monospace, monospace"
      );
      if (revealed && pole.moved > 0.02) {
        pushLine(
          [pole.orig[0], pole.orig[1], 0],
          [pole.orig[0], pole.orig[1], Math.max(pole.orig[2], 0)],
          COLORS.nominalGhost,
          1,
          [3, 4]
        );
        pushLine(pole.orig, pole.truth, COLORS.nominalGhost, 1, [2, 4]);
      }
    }
  }

  /**
   * The cylinder inside which true positions are revealed, drawn at the drone's altitude.
   *
   * @param {Object} model Replay model.
   * @param {number} step Step being shown.
   * @param {Function} pushPoly Polyline primitive helper.
   */
  function drawSensorRange(model, step, pushPoly) {
    if (!(model.sensorRange > 0)) return;
    const point = model.pos[step];
    const ring = Array.from({ length: 48 }, (_, i) => {
      const angle = (i / 48) * Math.PI * 2;
      return [
        point[0] + model.sensorRange * Math.cos(angle),
        point[1] + model.sensorRange * Math.sin(angle),
        point[2],
      ];
    });
    pushPoly(ring, COLORS.sensor, 1, [4, 4]);
  }

  /**
   * The drone: body axes, heading, speed colour, and a drop line to the floor for altitude.
   *
   * @param {Object} model Replay model.
   * @param {number} step Step being shown.
   * @param {Function} pushLine Line primitive helper.
   * @param {Function} pushPoint Point primitive helper.
   * @param {Function} colorFor Speed to colour helper.
   */
  function drawDrone(model, step, pushLine, pushPoint, colorFor) {
    const point = model.pos[step];
    const quat = model.quat ? model.quat[step] : [0, 0, 0, 1];
    const color = colorFor(model.speedFraction(model.speed[step]));
    const bodyX = rotate(quat, [1, 0, 0]);
    const bodyY = rotate(quat, [0, 1, 0]);
    const bodyZ = rotate(quat, [0, 0, 1]);

    pushLine([point[0], point[1], 0], point, "rgba(255, 255, 255, 0.22)", 1, [2, 3]);
    pushLine(add(point, scale(bodyX, 0.09)), sub(point, scale(bodyX, 0.09)), COLORS.drone, 2.2, null);
    pushLine(add(point, scale(bodyY, 0.09)), sub(point, scale(bodyY, 0.09)), COLORS.drone, 2.2, null);
    pushLine(add(point, scale(bodyZ, 0.06)), sub(point, scale(bodyZ, 0.06)), COLORS.drone, 1.8, null);
    pushLine(point, add(point, scale(bodyX, 0.3)), color, 3, null);
    pushPoint(point, 5, color);
  }

  /**
   * Orbit, zoom and reset with the mouse, touch or pen.
   *
   * @param {HTMLCanvasElement} canvas Canvas to listen on.
   * @param {Object} state App state holding `camera` and `model`.
   * @param {Function} onChange Called whenever the camera moved, to request a redraw.
   */
  function attachControls(canvas, state, onChange) {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const camera = state.camera;
      camera.azimuth -= (event.clientX - lastX) * 0.008;
      camera.elevation += (event.clientY - lastY) * 0.008;
      camera.elevation = Math.min(1.4, Math.max(-0.25, camera.elevation));
      lastX = event.clientX;
      lastY = event.clientY;
      onChange();
    });
    const stop = (event) => {
      if (!dragging) return;
      dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      canvas.classList.remove("dragging");
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const camera = state.camera;
        camera.distance = Math.min(24, Math.max(1.2, camera.distance * Math.exp(event.deltaY * 0.0012)));
        onChange();
      },
      { passive: false }
    );
    canvas.addEventListener("dblclick", () => {
      if (!state.model) return;
      state.camera = defaultCamera(state.model);
      onChange();
    });
  }

  NS.view3d = { draw, attachControls, defaultCamera };
})(typeof globalThis !== "undefined" ? globalThis : this);
