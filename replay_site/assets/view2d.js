/**
 * The 2D view: the track from above, with the flown trajectory coloured by speed.
 *
 * The view is deliberately the same picture as the PNG that `scripts/debug_run.py` writes, so a run
 * can be compared between the two: same geometry, same speed ramp, same nominal-versus-true
 * distinction. There is no camera to lose track of, which makes this the view that answers "where
 * did it go".
 *
 * Exposed as `Replay.view2d.draw(canvas, model, state)`.
 */
(function (root) {
  "use strict";

  const NS = (root.Replay = root.Replay || {});
  const {
    GEOMETRY,
    speedColor,
    rotate,
    yawOf,
    prepareCanvas,
    label,
  } = NS.util;

  const COLORS = {
    background: "#0f151d",
    grid: "rgba(140, 170, 210, 0.07)",
    gridStrong: "rgba(140, 170, 210, 0.16)",
    tick: "rgba(150, 170, 195, 0.55)",
    future: "rgba(190, 205, 225, 0.12)",
    gate: "#e8a33d",
    gateDim: "rgba(232, 163, 61, 0.42)",
    gateOpening: "rgba(200, 200, 200, 0.35)",
    pole: "#b388ff",
    poleDim: "rgba(179, 136, 255, 0.45)",
    contact: "rgba(255, 100, 100, 0.55)",
    sensor: "rgba(80, 200, 255, 0.45)",
    nominalGhost: "rgba(160, 175, 195, 0.45)",
    drone: "#ffffff",
    text: "#d7dde8",
  };

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
   * @param {Object} model Replay model from `Replay.model.build`.
   * @param {{step: number, revealedAll: boolean}} state What to show.
   * @returns {{x: Function, y: Function, scale: number}} The world-to-screen transform, so the app
   *     can answer clicks on the canvas.
   */
  function draw(canvas, model, state) {
    const { ctx, width, height } = prepareCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    const view = model.view;
    const padding = 26;
    const scale = Math.min(
      (width - 2 * padding) / (view.xmax - view.xmin),
      (height - 2 * padding) / (view.ymax - view.ymin)
    );
    const offsetX = padding + ((width - 2 * padding) - (view.xmax - view.xmin) * scale) / 2;
    const offsetY = padding + ((height - 2 * padding) - (view.ymax - view.ymin) * scale) / 2;
    const screenX = (x) => offsetX + (x - view.xmin) * scale;
    const screenY = (y) => offsetY + (view.ymax - y) * scale;

    drawGrid(ctx, model, width, height, screenX, screenY, scale);
    drawFuture(ctx, model, screenX, screenY);
    for (const gate of model.gates) drawGate(ctx, gate, state, screenX, screenY, scale);
    for (const pole of model.poles) drawPole(ctx, pole, state, screenX, screenY, scale);

    drawTrail(ctx, model, state.step, screenX, screenY);

    const step = state.step;
    const position = model.pos[step];
    drawSensorRange(ctx, model, step, screenX, screenY, scale);
    drawDrone(ctx, model, step, screenX, screenY);

    const axes = { x: screenX, y: screenY, scale };
    ctx.save();
    label(ctx, "x [m]", width - 12, height - 12, { align: "right", color: COLORS.tick });
    label(ctx, "y [m]", 12, 14, { color: COLORS.tick });
    ctx.restore();
    return { screenX, screenY, scale, width, height, position };
  }

  /**
   * Metre grid and its tick labels.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {Object} model Replay model.
   * @param {number} width Canvas width in CSS pixels.
   * @param {number} height Canvas height in CSS pixels.
   * @param {Function} screenX World x to screen x.
   * @param {Function} screenY World y to screen y.
   * @param {number} scale Pixels per metre.
   */
  function drawGrid(ctx, model, width, height, screenX, screenY, scale) {
    const view = model.view;
    ctx.save();
    ctx.lineWidth = 1;
    for (let x = Math.ceil(view.xmin); x <= view.xmax; x++) {
      ctx.strokeStyle = x === 0 ? COLORS.gridStrong : COLORS.grid;
      ctx.beginPath();
      ctx.moveTo(screenX(x), screenY(view.ymin));
      ctx.lineTo(screenX(x), screenY(view.ymax));
      ctx.stroke();
      if (scale > 40) {
        label(ctx, String(x), screenX(x) + 3, height - 22, { color: COLORS.tick, font: "9px ui-monospace, monospace" });
      }
    }
    for (let y = Math.ceil(view.ymin); y <= view.ymax; y++) {
      ctx.strokeStyle = y === 0 ? COLORS.gridStrong : COLORS.grid;
      ctx.beginPath();
      ctx.moveTo(screenX(view.xmin), screenY(y));
      ctx.lineTo(screenX(view.xmax), screenY(y));
      ctx.stroke();
      if (scale > 40) {
        label(ctx, String(y), 6, screenY(y) - 6, { color: COLORS.tick, font: "9px ui-monospace, monospace" });
      }
    }
    ctx.restore();
  }

  /**
   * The whole trajectory, faint, so the viewer knows where the flight is heading before it gets there.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {Object} model Replay model.
   * @param {Function} screenX World x to screen x.
   * @param {Function} screenY World y to screen y.
   */
  function drawFuture(ctx, model, screenX, screenY) {
    if (model.n < 2) return;
    ctx.save();
    ctx.strokeStyle = COLORS.future;
    ctx.lineWidth = 1;
    ctx.beginPath();
    model.pos.forEach((point, index) => {
      const x = screenX(point[0]);
      const y = screenY(point[1]);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }

  /**
   * The trajectory flown so far, one coloured segment per step.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {Object} model Replay model.
   * @param {number} step Last step to draw.
   * @param {Function} screenX World x to screen x.
   * @param {Function} screenY World y to screen y.
   */
  function drawTrail(ctx, model, step, screenX, screenY) {
    if (step < 1) return;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineWidth = 2.8;
    for (let i = 1; i <= step; i++) {
      ctx.strokeStyle = speedColor(model.speedFraction(model.speed[i]));
      ctx.beginPath();
      ctx.moveTo(screenX(model.pos[i - 1][0]), screenY(model.pos[i - 1][1]));
      ctx.lineTo(screenX(model.pos[i][0]), screenY(model.pos[i][1]));
      ctx.stroke();
    }
    ctx.restore();
  }

  /**
   * A gate: its frame seen from above (the two side posts), its opening and its crossing direction.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {Object} gate Gate model.
   * @param {{step: number, revealedAll: boolean}} state What to show.
   * @param {Function} screenX World x to screen x.
   * @param {Function} screenY World y to screen y.
   * @param {number} scale Pixels per metre.
   */
  function drawGate(ctx, gate, state, screenX, screenY, scale) {
    const revealed = isRevealed(gate, state.step, state.revealedAll);
    const centre = revealed ? gate.truth : gate.orig;
    const quat = revealed ? gate.truthQuat : gate.quat;
    const lateral = rotate(quat, [0, 1, 0]);
    const normal = rotate(quat, [1, 0, 0]);
    const color = revealed ? COLORS.gate : COLORS.gateDim;

    ctx.save();
    ctx.setLineDash(revealed ? [] : [4, 4]);
    // The frame is a square ring; from above only the two side posts are visible. The gap between
    // them is the opening the drone has to cross.
    ctx.strokeStyle = color;
    ctx.lineCap = "butt";
    ctx.lineWidth = revealed ? 5 : 3;
    for (const sign of [1, -1]) {
      ctx.beginPath();
      ctx.moveTo(
        screenX(centre[0] + sign * GEOMETRY.gateOpeningHalf * lateral[0]),
        screenY(centre[1] + sign * GEOMETRY.gateOpeningHalf * lateral[1])
      );
      ctx.lineTo(
        screenX(centre[0] + sign * GEOMETRY.gateFrameHalf * lateral[0]),
        screenY(centre[1] + sign * GEOMETRY.gateFrameHalf * lateral[1])
      );
      ctx.stroke();
    }
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = COLORS.gateOpening;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(
      screenX(centre[0] - GEOMETRY.gateOpeningHalf * lateral[0]),
      screenY(centre[1] - GEOMETRY.gateOpeningHalf * lateral[1])
    );
    ctx.lineTo(
      screenX(centre[0] + GEOMETRY.gateOpeningHalf * lateral[0]),
      screenY(centre[1] + GEOMETRY.gateOpeningHalf * lateral[1])
    );
    ctx.stroke();
    ctx.restore();

    // Crossing direction, as an arrow through the gate.
    arrow(
      ctx,
      screenX(centre[0]),
      screenY(centre[1]),
      screenX(centre[0] + 0.3 * normal[0]),
      screenY(centre[1] + 0.3 * normal[1]),
      color
    );

    // Nominal position, once the true one is known and they differ: the nominal track is what the
    // controller planned on before the sensor revealed the real gate.
    if (revealed && gate.moved > 0.02) {
      ctx.save();
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = COLORS.nominalGhost;
      ctx.lineWidth = 1.2;
      for (const sign of [1, -1]) {
        const lateralNominal = rotate(gate.quat, [0, 1, 0]);
        ctx.beginPath();
        ctx.moveTo(
          screenX(gate.orig[0] + sign * GEOMETRY.gateOpeningHalf * lateralNominal[0]),
          screenY(gate.orig[1] + sign * GEOMETRY.gateOpeningHalf * lateralNominal[1])
        );
        ctx.lineTo(
          screenX(gate.orig[0] + sign * GEOMETRY.gateFrameHalf * lateralNominal[0]),
          screenY(gate.orig[1] + sign * GEOMETRY.gateFrameHalf * lateralNominal[1])
        );
        ctx.stroke();
      }
      ctx.strokeStyle = "rgba(160, 175, 195, 0.30)";
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(screenX(gate.orig[0]), screenY(gate.orig[1]));
      ctx.lineTo(screenX(gate.truth[0]), screenY(gate.truth[1]));
      ctx.stroke();
      ctx.restore();
      label(ctx, `g${gate.id} nominal`, screenX(gate.orig[0]) + 8, screenY(gate.orig[1]) + 12, {
        color: COLORS.nominalGhost,
        font: "9px ui-monospace, monospace",
      });
    }

    label(
      ctx,
      `g${gate.id}${revealed ? "" : " ?"}`,
      screenX(centre[0]) + 9,
      screenY(centre[1]) - 9,
      { color, font: "bold 11px ui-monospace, monospace" }
    );
  }

  /**
   * A pole: a vertical capsule, so from above it is a dot with the radius at which the drone touches
   * it.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {Object} pole Pole model.
   * @param {{step: number, revealedAll: boolean}} state What to show.
   * @param {Function} screenX World x to screen x.
   * @param {Function} screenY World y to screen y.
   * @param {number} scale Pixels per metre.
   */
  function drawPole(ctx, pole, state, screenX, screenY, scale) {
    const revealed = isRevealed(pole, state.step, state.revealedAll);
    const centre = revealed ? pole.truth : pole.orig;
    const x = screenX(centre[0]);
    const y = screenY(centre[1]);
    const contact = GEOMETRY.poleRadius + GEOMETRY.droneRadius;

    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = revealed ? COLORS.contact : "rgba(255, 100, 100, 0.28)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(x, y, contact * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(2, GEOMETRY.poleRadius * scale), 0, Math.PI * 2);
    if (revealed) {
      ctx.fillStyle = COLORS.pole;
      ctx.fill();
    } else {
      ctx.fillStyle = "rgba(15, 21, 29, 0.9)";
      ctx.fill();
      ctx.strokeStyle = COLORS.poleDim;
      ctx.stroke();
    }
    ctx.restore();

    if (revealed && pole.moved > 0.02) {
      ctx.save();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = "rgba(160, 175, 195, 0.30)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(screenX(pole.orig[0]), screenY(pole.orig[1]));
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(255, 100, 100, 0.22)";
      ctx.beginPath();
      ctx.arc(screenX(pole.orig[0]), screenY(pole.orig[1]), contact * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    label(ctx, `o${pole.id}${revealed ? "" : " ?"}`, x + 7, y + 12, {
      color: revealed ? COLORS.pole : COLORS.poleDim,
      font: "10px ui-monospace, monospace",
    });
  }

  /**
   * The circle inside which the environment reveals true gate and pole positions.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {Object} model Replay model.
   * @param {number} step Step being shown.
   * @param {Function} screenX World x to screen x.
   * @param {Function} screenY World y to screen y.
   * @param {number} scale Pixels per metre.
   */
  function drawSensorRange(ctx, model, step, screenX, screenY, scale) {
    if (!(model.sensorRange > 0)) return;
    const point = model.pos[step];
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = COLORS.sensor;
    ctx.fillStyle = "rgba(80, 200, 255, 0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(screenX(point[0]), screenY(point[1]), model.sensorRange * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  /**
   * The drone at the current step: a heading arrow coloured by its instantaneous speed.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {Object} model Replay model.
   * @param {number} step Step being shown.
   * @param {Function} screenX World x to screen x.
   * @param {Function} screenY World y to screen y.
   */
  function drawDrone(ctx, model, step, screenX, screenY) {
    const point = model.pos[step];
    const x = screenX(point[0]);
    const y = screenY(point[1]);
    const yaw = model.quat ? yawOf(model.quat[step]) : 0;
    const color = speedColor(model.speedFraction(model.speed[step]));
    ctx.save();
    ctx.translate(x, y);
    // The screen's y axis points down, so a world yaw is a clockwise rotation here.
    ctx.rotate(-yaw);
    ctx.beginPath();
    ctx.moveTo(11, 0);
    ctx.lineTo(-6, 6);
    ctx.lineTo(-3, 0);
    ctx.lineTo(-6, -6);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = COLORS.drone;
    ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, 1.6, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.drone;
    ctx.fill();
    ctx.restore();
  }

  /**
   * A line with an arrow head, used for the gate crossing direction.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {number} x0 Tail x in screen pixels.
   * @param {number} y0 Tail y in screen pixels.
   * @param {number} x1 Head x in screen pixels.
   * @param {number} y1 Head y in screen pixels.
   * @param {string} color Stroke colour.
   */
  function arrow(ctx, x0, y0, x1, y1, color) {
    const angle = Math.atan2(y1 - y0, x1 - x0);
    const head = 7;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(angle - 0.4), y1 - head * Math.sin(angle - 0.4));
    ctx.lineTo(x1 - head * Math.cos(angle + 0.4), y1 - head * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
  }

  NS.view2d = { draw };
})(typeof globalThis !== "undefined" ? globalThis : this);
