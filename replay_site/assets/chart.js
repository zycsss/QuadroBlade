/**
 * The time charts under the views: altitude and speed over the whole run, with the playback position
 * marked on both.
 *
 * These are the same two quantities the debug figure draws: the height (§ altitude panel) and the
 * speed that colours the trajectory. Showing them against the full run is what turns the replay from
 * an animation into a diagnostic - a dip in the altitude trace at the exact moment of a gate pass is
 * visible without scrubbing.
 *
 * Exposed as `Replay.chart.draw(canvas, model, state)`.
 */
(function (root) {
  "use strict";

  const NS = (root.Replay = root.Replay || {});
  const { GEOMETRY, speedColor, prepareCanvas, label } = NS.util;

  const COLORS = {
    background: "#0d131b",
    panel: "rgba(255, 255, 255, 0.015)",
    border: "rgba(140, 170, 210, 0.12)",
    grid: "rgba(140, 170, 210, 0.10)",
    tick: "rgba(150, 170, 195, 0.70)",
    commanded: "rgba(205, 215, 230, 0.50)",
    gate: "rgba(240, 173, 78, 0.55)",
    cursor: "#ffd166",
  };

  /**
   * Draw one frame.
   *
   * @param {HTMLCanvasElement} canvas Canvas to draw on.
   * @param {Object} model Replay model.
   * @param {{step: number}} state Playback state.
   * @returns {{stepAt: Function}} Mapping from a canvas x coordinate to a step, for click-to-seek.
   */
  function draw(canvas, model, state) {
    const { ctx, width, height } = prepareCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    const paddingLeft = 50;
    const paddingRight = 14;
    const paddingTop = 14;
    const paddingBottom = 20;
    const plotWidth = Math.max(10, width - paddingLeft - paddingRight);
    const plotHeight = Math.max(20, height - paddingTop - paddingBottom);
    const gap = 18;
    const altitudeHeight = Math.max(20, (plotHeight - gap) * 0.64);
    const speedHeight = Math.max(16, plotHeight - gap - altitudeHeight);
    const altitudeTop = paddingTop;
    const speedTop = paddingTop + altitudeHeight + gap;

    const timeStart = model.t[0];
    const timeEnd = model.t[model.n - 1];
    const timeSpan = timeEnd - timeStart || 1;
    const xOfTime = (time) => paddingLeft + ((time - timeStart) / timeSpan) * plotWidth;

    const altitude = altitudeDomain(model);
    const yOfAltitude = (z) =>
      altitudeTop + altitudeHeight - ((z - altitude.low) / (altitude.high - altitude.low)) * altitudeHeight;

    const speedHigh = Math.max(1, model.speedMax) * 1.05;
    const yOfSpeed = (value) => speedTop + speedHeight - (value / speedHigh) * speedHeight;

    ctx.save();
    drawPanel(ctx, paddingLeft, altitudeTop, plotWidth, altitudeHeight);
    drawPanel(ctx, paddingLeft, speedTop, plotWidth, speedHeight);

    // Horizontal grid and the numbers on the left.
    for (const z of altitudeTicks(altitude)) {
      const y = yOfAltitude(z);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, y);
      ctx.lineTo(paddingLeft + plotWidth, y);
      ctx.stroke();
      label(ctx, z.toFixed(2), paddingLeft - 6, y, {
        align: "right",
        color: COLORS.tick,
        font: "9px ui-monospace, monospace",
      });
    }
    for (const value of [0, speedHigh / 2, speedHigh]) {
      const y = yOfSpeed(value);
      ctx.strokeStyle = COLORS.grid;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, y);
      ctx.lineTo(paddingLeft + plotWidth, y);
      ctx.stroke();
      label(ctx, value.toFixed(1), paddingLeft - 6, y, {
        align: "right",
        color: COLORS.tick,
        font: "9px ui-monospace, monospace",
      });
    }

    // Gate passes: a faint time guide across both panels, and the height of the gate's opening drawn
    // as an interval at the top. The interval is what the drone had to be inside; its centre alone
    // said nothing about whether a pass through the opening was even possible.
    for (const pass of model.passes) {
      const x = xOfTime(pass.t);
      ctx.strokeStyle = "rgba(240, 173, 78, 0.28)";
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, altitudeTop);
      ctx.lineTo(x, altitudeTop + altitudeHeight);
      ctx.moveTo(x, speedTop);
      ctx.lineTo(x, speedTop + speedHeight);
      ctx.stroke();
      ctx.setLineDash([]);

      const gate = model.gates[pass.gate];
      if (!gate) continue;
      const centre = gate.truth[2];
      const low = yOfAltitude(centre - GEOMETRY.gateOpeningHalf);
      const high = yOfAltitude(centre + GEOMETRY.gateOpeningHalf);
      // The frame, thin, then the opening it leaves, as a bar with caps at both ends.
      ctx.strokeStyle = "rgba(240, 173, 78, 0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, yOfAltitude(centre - GEOMETRY.gateFrameHalf));
      ctx.lineTo(x, yOfAltitude(centre + GEOMETRY.gateFrameHalf));
      ctx.stroke();
      ctx.strokeStyle = COLORS.gate;
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(x, low);
      ctx.lineTo(x, high);
      ctx.stroke();
      ctx.lineWidth = 1.4;
      for (const y of [low, high]) {
        ctx.beginPath();
        ctx.moveTo(x - 5, y);
        ctx.lineTo(x + 5, y);
        ctx.stroke();
      }
      label(ctx, `g${pass.gate}`, x + 6, Math.max(altitudeTop + 8, high - 10), {
        color: COLORS.gate,
        font: "bold 9.5px ui-monospace, monospace",
      });
    }

    // Commanded height, which the controller asked for, against the height it got. Only present when
    // the controller commanded a position (state mode); an attitude command has no height in it.
    if (model.commandedHeight) {
      ctx.strokeStyle = COLORS.commanded;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      for (let i = 0; i < model.n; i++) {
        const x = xOfTime(model.t[i]);
        const y = yOfAltitude(model.commandedHeight[i][2]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // The altitude trace, coloured by speed like the trajectory it belongs to.
    ctx.lineCap = "round";
    ctx.lineWidth = 2.2;
    for (let i = 1; i < model.n; i++) {
      ctx.strokeStyle = speedColor(model.speedFraction(model.speed[i]));
      ctx.beginPath();
      ctx.moveTo(xOfTime(model.t[i - 1]), yOfAltitude(model.pos[i - 1][2]));
      ctx.lineTo(xOfTime(model.t[i]), yOfAltitude(model.pos[i][2]));
      ctx.stroke();
    }

    // Speed trace, same colour code.
    ctx.lineWidth = 1.8;
    for (let i = 1; i < model.n; i++) {
      ctx.strokeStyle = speedColor(model.speedFraction(model.speed[i]));
      ctx.beginPath();
      ctx.moveTo(xOfTime(model.t[i - 1]), yOfSpeed(model.speed[i - 1]));
      ctx.lineTo(xOfTime(model.t[i]), yOfSpeed(model.speed[i]));
      ctx.stroke();
    }

    // Playback cursor.
    const step = Math.min(model.n - 1, Math.max(0, state.step));
    const cursorX = xOfTime(model.t[step]);
    ctx.strokeStyle = COLORS.cursor;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cursorX, altitudeTop);
    ctx.lineTo(cursorX, altitudeTop + altitudeHeight);
    ctx.moveTo(cursorX, speedTop);
    ctx.lineTo(cursorX, speedTop + speedHeight);
    ctx.stroke();
    for (const y of [yOfAltitude(model.pos[step][2]), yOfSpeed(model.speed[step])]) {
      ctx.fillStyle = COLORS.cursor;
      ctx.beginPath();
      ctx.arc(cursorX, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    label(ctx, "z [m]", paddingLeft + 4, altitudeTop + 9, { color: COLORS.tick, font: "10px ui-monospace, monospace" });
    label(ctx, "speed [m/s]", paddingLeft + 4, speedTop + 9, { color: COLORS.tick, font: "10px ui-monospace, monospace" });
    for (let time = Math.ceil(timeStart); time <= timeEnd; time++) {
      const x = xOfTime(time);
      label(ctx, `${time}`, x, height - 8, {
        align: "center",
        color: COLORS.tick,
        font: "9px ui-monospace, monospace",
      });
    }
    label(ctx, "time [s]", width - paddingRight, height - 8, { align: "right", color: COLORS.tick, font: "9px ui-monospace, monospace" });

    // The readings at the cursor: the vertical line the mouse is on, the two traces it crosses, and
    // the numbers there. Hovering only previews; clicking still seeks the playback.
    if (state.hoverChart && Number.isFinite(state.hoverChart.x) && plotWidth > 0 && timeSpan > 0) {
      const hoverX = Math.min(paddingLeft + plotWidth, Math.max(paddingLeft, state.hoverChart.x));
      const time = timeStart + ((hoverX - paddingLeft) / plotWidth) * timeSpan;
      const index = nearestStep(model, time);
      const z = model.pos[index][2];
      const speed = model.speed[index];

      ctx.save();
      ctx.strokeStyle = "rgba(230, 240, 255, 0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, altitudeTop);
      ctx.lineTo(hoverX, speedTop + speedHeight);
      ctx.stroke();
      for (const [y, color] of [
        [yOfAltitude(z), speedColor(model.speedFraction(speed))],
        [yOfSpeed(speed), speedColor(model.speedFraction(speed))],
      ]) {
        ctx.beginPath();
        ctx.arc(hoverX, y, 3.6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.3;
        ctx.stroke();
      }
      ctx.restore();

      drawTooltip(
        ctx,
        hoverX,
        Math.min(speedTop + speedHeight, Math.max(altitudeTop, state.hoverChart.y)),
        [
          { text: `t ${model.t[index].toFixed(2)} s`, color: "#e9eef7" },
          { text: `z ${z.toFixed(2)} m`, color: "#e9eef7" },
          { text: `speed ${speed.toFixed(2)} m/s`, color: speedColor(model.speedFraction(speed)) },
        ],
        width
      );
    }
    ctx.restore();

    return {
      stepAt: (canvasX) => {
        const fraction = (canvasX - paddingLeft) / (plotWidth || 1);
        return Math.round(Math.min(1, Math.max(0, fraction)) * (model.n - 1));
      },
    };
  }

  /**
   * Index of the logged step closest to a time. The times are sorted, so this is a binary search.
   *
   * @param {Object} model Replay model.
   * @param {number} time Time in seconds.
   * @returns {number} Step index.
   */
  function nearestStep(model, time) {
    let low = 0;
    let high = model.n - 1;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (model.t[middle] < time) low = middle + 1;
      else high = middle;
    }
    const after = low;
    const before = Math.max(0, after - 1);
    return Math.abs(model.t[before] - time) <= Math.abs(model.t[after] - time) ? before : after;
  }

  /**
   * A small dark box with a few lines of text, kept inside the canvas and flipped away from the edge.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {number} x Anchor x in CSS pixels.
   * @param {number} y Anchor y in CSS pixels.
   * @param {Array<{text: string, color: string}>} lines Lines to draw.
   * @param {number} width Canvas width in CSS pixels.
   */
  function drawTooltip(ctx, x, y, lines, width) {
    ctx.save();
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    const textWidth = Math.max(...lines.map((line) => ctx.measureText(line.text).width));
    const boxWidth = textWidth + 16;
    const boxHeight = lines.length * 15 + 8;
    const left = x + 14 + boxWidth > width - 4 ? x - 14 - boxWidth : x + 14;
    const top = y - boxHeight - 10 < 4 ? y + 14 : y - boxHeight - 10;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") ctx.roundRect(left, top, boxWidth, boxHeight, 5);
    else ctx.rect(left, top, boxWidth, boxHeight);
    ctx.fillStyle = "rgba(10, 15, 22, 0.94)";
    ctx.fill();
    ctx.strokeStyle = "rgba(140, 170, 210, 0.4)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    lines.forEach((line, index) => {
      ctx.fillStyle = line.color;
      ctx.fillText(line.text, left + 8, top + 5 + index * 15);
    });
    ctx.restore();
  }

  /**
   * Height range that has to fit in the altitude panel.
   *
   * @param {Object} model Replay model.
   * @returns {{low: number, high: number}} Range in metres.
   */
  function altitudeDomain(model) {
    let low = 0;
    let high = 0;
    const extend = (z) => {
      low = Math.min(low, z);
      high = Math.max(high, z);
    };
    for (const point of model.pos) extend(point[2]);
    if (model.commandedHeight) for (const point of model.commandedHeight) extend(point[2]);
    // The gate's whole frame has to fit, not just its centre: the pass is drawn as an interval.
    for (const gate of model.gates) {
      extend(gate.truth[2] - GEOMETRY.gateFrameHalf);
      extend(gate.truth[2] + GEOMETRY.gateFrameHalf);
    }
    low -= 0.08;
    high += 0.08;
    if (high - low < 0.25) high = low + 0.25;
    return { low, high };
  }

  /**
   * Three round tick heights inside the altitude range.
   *
   * @param {{low: number, high: number}} domain Altitude range.
   * @returns {number[]} Tick values in metres.
   */
  function altitudeTicks(domain) {
    const span = domain.high - domain.low;
    const step = span > 1.6 ? 0.5 : span > 0.8 ? 0.25 : 0.1;
    const ticks = [];
    for (let value = Math.ceil(domain.low / step) * step; value <= domain.high + 1e-9; value += step) {
      ticks.push(Number(value.toFixed(3)));
    }
    return ticks.length ? ticks : [domain.low, domain.high];
  }

  /**
   * Panel background and border.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {number} x Left edge in CSS pixels.
   * @param {number} y Top edge in CSS pixels.
   * @param {number} width Panel width.
   * @param {number} height Panel height.
   */
  function drawPanel(ctx, x, y, width, height) {
    ctx.fillStyle = COLORS.panel;
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
  }

  NS.chart = { draw };
})(typeof globalThis !== "undefined" ? globalThis : this);
