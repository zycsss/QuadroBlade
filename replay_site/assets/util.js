/**
 * Small helpers shared by the two views and the chart: scene geometry, quaternions, the speed colour
 * ramp and canvas sizing.
 *
 * Exposed as `Replay.util`.
 */
(function (root) {
  "use strict";

  const NS = (root.Replay = root.Replay || {});

  /**
   * Geometry of the track, copied from the simulation so the replay is drawn at true scale.
   *
   * `droneRadius`, `poleRadius`, `gateFrameHalf` and `gateOpeningHalf` match
   * `lsy_drone_racing/control/astar.py`; `poleHeight` matches `envs/assets/obstacle.xml`, where the
   * capsule hangs 1.6 m below the configured pole position (which is therefore its top).
   */
  const GEOMETRY = {
    droneRadius: 0.107,
    poleRadius: 0.015,
    poleHeight: 1.6,
    gateFrameHalf: 0.36,
    gateOpeningHalf: 0.2,
  };

  /**
   * Stops of the speed ramp, the same blue-to-red reading as matplotlib's `turbo` used by the PNG
   * that `scripts/debug_run.py` writes, so a run looks the same in both places.
   */
  const SPEED_RAMP = [
    [0.0, [48, 18, 59]],
    [0.13, [62, 66, 176]],
    [0.27, [34, 120, 214]],
    [0.4, [28, 176, 205]],
    [0.53, [53, 206, 132]],
    [0.66, [160, 220, 60]],
    [0.79, [242, 205, 38]],
    [0.9, [250, 120, 30]],
    [1.0, [176, 20, 20]],
  ];

  /**
   * Colour of a speed, on the shared 0..1 ramp.
   *
   * @param {number} fraction Speed as a fraction of the fastest speed of the run, clamped to 0..1.
   * @returns {string} CSS `rgb()` colour.
   */
  function speedColor(fraction) {
    const f = Math.min(1, Math.max(0, Number.isFinite(fraction) ? fraction : 0));
    for (let i = 1; i < SPEED_RAMP.length; i++) {
      const [stop, rgb] = SPEED_RAMP[i];
      if (f <= stop) {
        const [prevStop, prevRgb] = SPEED_RAMP[i - 1];
        const span = stop - prevStop || 1;
        const k = (f - prevStop) / span;
        const mix = (a, b) => Math.round(a + (b - a) * k);
        return `rgb(${mix(prevRgb[0], rgb[0])},${mix(prevRgb[1], rgb[1])},${mix(prevRgb[2], rgb[2])})`;
      }
    }
    const last = SPEED_RAMP[SPEED_RAMP.length - 1][1];
    return `rgb(${last[0]},${last[1]},${last[2]})`;
  }

  /**
   * A CSS gradient that draws the whole ramp, for the legend.
   *
   * @returns {string} CSS `linear-gradient(...)`, left = slow, right = fast.
   */
  function speedGradientCss() {
    const stops = SPEED_RAMP.map(([stop, rgb]) => `rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${stop * 100}%`);
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  }

  /**
   * Rotate a vector by an xyzw quaternion.
   *
   * @param {number[]} q Quaternion in (x, y, z, w) order, as the log stores it.
   * @param {number[]} v Vector to rotate.
   * @returns {number[]} Rotated vector.
   */
  function rotate(q, v) {
    const [x, y, z, w] = q;
    const ix = w * v[0] + y * v[2] - z * v[1];
    const iy = w * v[1] + z * v[0] - x * v[2];
    const iz = w * v[2] + x * v[1] - y * v[0];
    const iw = -x * v[0] - y * v[1] - z * v[2];
    return [
      ix * w + iw * -x + iy * -z - iz * -y,
      iy * w + iw * -y + iz * -x - ix * -z,
      iz * w + iw * -z + ix * -y - iy * -x,
    ];
  }

  /**
   * Heading of a body, in the world's xy plane.
   *
   * @param {number[]} q Quaternion in (x, y, z, w) order.
   * @returns {number} Yaw in radians, 0 along +x.
   */
  function yawOf(q) {
    const [x, y, z, w] = q;
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  /**
   * Size a canvas to its CSS box at the device pixel ratio and return a ready 2D context.
   *
   * @param {HTMLCanvasElement} canvas Canvas element.
   * @returns {{ctx: CanvasRenderingContext2D, width: number, height: number}} Context and CSS size.
   */
  function prepareCanvas(canvas) {
    const ratio = Math.min(root.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth));
    const height = Math.max(1, Math.round(canvas.clientHeight));
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height };
  }

  /**
   * Draw text with a dark halo so it stays readable over the track.
   *
   * @param {CanvasRenderingContext2D} ctx Context.
   * @param {string} text Text to draw.
   * @param {number} x Position in CSS pixels.
   * @param {number} y Position in CSS pixels.
   * @param {{font?: string, color?: string, align?: string, baseline?: string}} [style] Style.
   */
  function label(ctx, text, x, y, style) {
    const options = style || {};
    ctx.save();
    ctx.font = options.font || "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = options.align || "left";
    ctx.textBaseline = options.baseline || "middle";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(8, 12, 18, 0.85)";
    ctx.strokeText(text, x, y);
    ctx.fillStyle = options.color || "#d7dde8";
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  // Minimal 3-vector arithmetic. Plain arrays keep the projection code short and readable.
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const length = (a) => Math.sqrt(dot(a, a));
  const normalize = (a) => {
    const len = length(a) || 1;
    return [a[0] / len, a[1] / len, a[2] / len];
  };
  const lerp3 = (a, b, t) => [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];

  NS.util = {
    GEOMETRY,
    SPEED_RAMP,
    speedColor,
    speedGradientCss,
    rotate,
    yawOf,
    prepareCanvas,
    label,
    add,
    sub,
    scale,
    dot,
    cross,
    length,
    normalize,
    lerp3,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
