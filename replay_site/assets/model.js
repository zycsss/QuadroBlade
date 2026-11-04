/**
 * Turn the arrays of one `.npz` run log into the model both views draw.
 *
 * Everything here comes from the log. The two positions of every object are the ones the log records:
 * `gates_pos`/`obstacles_pos` are the nominal track the controller started from, and
 * `true_gates_pos`/`true_obstacles_pos` are where the randomisation actually put them. The step at
 * which each object stops being nominal and starts being true is read from the logged
 * `gates_visited`/`poles_visited` flags; for a log written before those were recorded, it is derived
 * from the trajectory and the sensor range with the environment's own rule (2D distance below the
 * range, permanently, once reached).
 *
 * Exposed as `Replay.model.build(arrays, summary, name)`.
 */
(function (root) {
  "use strict";

  const NS = (root.Replay = root.Replay || {});
  const { GEOMETRY } = NS.util;

  /**
   * Read a value that may be a scalar, a one-element array or a 0-dimensional numpy array.
   *
   * @param {*} value Value from the log.
   * @param {number} fallback Value to use when it is missing.
   * @returns {number} The scalar.
   */
  function scalar(value, fallback) {
    if (value === undefined || value === null) return fallback;
    if (Array.isArray(value)) return value.length ? scalar(value[0], fallback) : fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  /**
   * Speed per step, for a log that does not store it.
   *
   * @param {number[][]} pos Positions, shape (n, 3).
   * @param {number[]} t Times, shape (n,).
   * @returns {number[]} Speed per step.
   */
  function deriveSpeed(pos, t) {
    return pos.map((p, i) => {
      if (i === 0) return 0;
      const dt = t[i] - t[i - 1] || 1;
      const d = [p[0] - pos[i - 1][0], p[1] - pos[i - 1][1], p[2] - pos[i - 1][2]];
      return Math.hypot(d[0], d[1], d[2]) / dt;
    });
  }

  /**
   * The step at which each object stops being nominal, using the environment's reveal rule.
   *
   * A logged flag is preferred when present, because that is what the controller was actually given.
   * The distance rule is the fallback and matches `_update_visited_objects` in
   * `lsy_drone_racing/envs/race_core.py`: horizontal distance below the sensor range, and once
   * revealed the object stays revealed.
   *
   * @param {number[][]} pos Drone positions, shape (n, 3).
   * @param {number[][]} truth True object positions, shape (m, 3).
   * @param {Array<number[]>|null} flags Per-step reveal flags, shape (n, m), or null.
   * @param {number} sensorRange Reveal distance in metres.
   * @returns {number[]} Step index per object, or -1 when it was never revealed.
   */
  function revealSteps(pos, truth, flags, sensorRange) {
    return truth.map((position, index) => {
      if (flags && flags.length) {
        for (let step = 0; step < flags.length; step++) {
          if (flags[step] && flags[step][index]) return step;
        }
        return -1;
      }
      if (!(sensorRange > 0)) return 0;
      for (let step = 0; step < pos.length; step++) {
        const dx = pos[step][0] - position[0];
        const dy = pos[step][1] - position[1];
        if (Math.hypot(dx, dy) < sensorRange) return step;
      }
      return -1;
    });
  }

  /**
   * Build one model per gate, holding both positions and when the true one became visible.
   *
   * @param {number[][]} orig Nominal positions, shape (m, 3).
   * @param {number[][]} truth True positions, shape (m, 3).
   * @param {number[][]} quat Nominal orientations as xyzw quaternions, shape (m, 4).
   * @param {number[][]} truthQuat True orientations, shape (m, 4).
   * @param {number[]} reveal Reveal step per object.
   * @returns {Array<Object>} Gate models.
   */
  function buildGates(orig, truth, quat, truthQuat, reveal) {
    return truth.map((position, index) => {
      const nominal = orig[index] || position;
      const orientation = quat[index] || [0, 0, 0, 1];
      return {
        id: index,
        orig: nominal,
        truth: position,
        quat: orientation,
        truthQuat: truthQuat[index] || orientation,
        reveal: reveal[index],
        moved: Math.hypot(
          nominal[0] - position[0],
          nominal[1] - position[1],
          nominal[2] - position[2]
        ),
      };
    });
  }

  /**
   * Build one model per pole. A pole has no orientation: it is a vertical capsule.
   *
   * @param {number[][]} orig Nominal positions, shape (m, 3).
   * @param {number[][]} truth True positions, shape (m, 3).
   * @param {number[]} reveal Reveal step per object.
   * @returns {Array<Object>} Pole models.
   */
  function buildPoles(orig, truth, reveal) {
    return truth.map((position, index) => {
      const nominal = orig[index] || position;
      return {
        id: index,
        orig: nominal,
        truth: position,
        reveal: reveal[index],
        moved: Math.hypot(
          nominal[0] - position[0],
          nominal[1] - position[1],
          nominal[2] - position[2]
        ),
        // The configured position is the top of the pole; the capsule hangs down from there.
        top: Math.max(position[2], 0),
        bottom: Math.max(0, position[2] - GEOMETRY.poleHeight),
      };
    });
  }

  /**
   * Gate passes, reconstructed from the step-by-step `n_gates_passed` counter.
   *
   * @param {number[]} passed Passing counter per step.
   * @param {number[]} t Times per step.
   * @param {number[]} sequence Gate id per required pass.
   * @returns {Array<{step: number, t: number, gate: number}>} One entry per completed pass.
   */
  function buildPasses(passed, t, sequence) {
    const passes = [];
    for (let step = 0; step < passed.length - 1; step++) {
      for (let k = passed[step]; k < passed[step + 1]; k++) {
        passes.push({ step: step + 1, t: t[step + 1], gate: sequence[k] === undefined ? k : sequence[k] });
      }
    }
    return passes;
  }

  /**
   * Build the replay model.
   *
   * @param {Object<string, *>} arrays Arrays of one `.npz` run log.
   * @param {Object|null} summary The `.json` summary written next to the log, or null.
   * @param {string} name File name of the log, used in the title when the summary is missing.
   * @returns {Object} The model.
   */
  function build(arrays, summary, name) {
    const t = arrays.t;
    const pos = arrays.pos;
    if (!Array.isArray(t) || !Array.isArray(pos) || !pos.length) {
      throw new Error("this log holds no trajectory (no `t` and `pos` arrays)");
    }
    const steps = pos.length;
    const quat = arrays.quat && arrays.quat.length === steps ? arrays.quat : null;
    const des = arrays.des && arrays.des.length === steps ? arrays.des : null;
    // The action space decides how `des` may be read. In state mode it is the commanded position; in
    // attitude mode it is roll/pitch/yaw, and plotting its third component as a height would be a
    // plausible-looking lie. Older logs record the mode only in the summary, or not at all.
    const controlMode = String(
      (arrays.control_mode !== undefined ? arrays.control_mode : (summary && summary.control_mode)) || ""
    ).toLowerCase();
    const commandedHeight = controlMode === "state" && des ? des : null;
    const passed = arrays.n && arrays.n.length === steps ? arrays.n : new Array(steps).fill(0);
    const speed =
      arrays.speed && arrays.speed.length === steps ? arrays.speed : deriveSpeed(pos, t);
    const sequence = Array.isArray(arrays.gate_sequence) ? arrays.gate_sequence : [];
    const sensorRange = scalar(arrays.sensor_range, scalar(summary && summary.sensor_range, 0.5));

    const gateOrig = arrays.gates_pos || arrays.true_gates_pos || [];
    const gateTruth = arrays.true_gates_pos || gateOrig;
    const gateQuat = arrays.gates_quat || [];
    const gateTruthQuat = arrays.true_gates_quat || gateQuat;
    const poleOrig = arrays.obstacles_pos || arrays.true_obstacles_pos || [];
    const poleTruth = arrays.true_obstacles_pos || poleOrig;

    const gates = buildGates(
      gateOrig,
      gateTruth,
      gateQuat,
      gateTruthQuat,
      revealSteps(pos, gateTruth, arrays.gates_visited, sensorRange)
    );
    const poles = buildPoles(
      poleOrig,
      poleTruth,
      revealSteps(pos, poleTruth, arrays.poles_visited, sensorRange)
    );

    // The view has to hold everything the replay can show, whether or not it is revealed yet.
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    let zmin = 0;
    let zmax = 1;
    const extend = (point) => {
      if (!point) return;
      xmin = Math.min(xmin, point[0]);
      xmax = Math.max(xmax, point[0]);
      ymin = Math.min(ymin, point[1]);
      ymax = Math.max(ymax, point[1]);
      zmin = Math.min(zmin, point[2]);
      zmax = Math.max(zmax, point[2]);
    };
    pos.forEach(extend);
    if (des) des.forEach(extend);
    for (const gate of gates) {
      extend(gate.orig);
      extend(gate.truth);
      zmin = Math.min(zmin, gate.truth[2] - GEOMETRY.gateFrameHalf);
      zmax = Math.max(zmax, gate.truth[2] + GEOMETRY.gateFrameHalf);
    }
    for (const pole of poles) {
      extend(pole.orig);
      extend(pole.truth);
      zmax = Math.max(zmax, pole.top);
    }
    if (!Number.isFinite(xmin)) {
      xmin = -1;
      xmax = 1;
      ymin = -1;
      ymax = 1;
    }
    const margin = 0.35;
    const view = {
      xmin: xmin - margin,
      xmax: xmax + margin,
      ymin: ymin - margin,
      ymax: ymax + margin,
    };

    const speedMin = Math.min(...speed);
    const speedMax = Math.max(...speed);
    const span = speedMax - speedMin;

    let dt = 0.02;
    if (steps > 1) {
      const deltas = t.slice(1).map((time, index) => time - t[index]).sort((a, b) => a - b);
      dt = deltas[Math.floor(deltas.length / 2)] || 0.02;
    }

    const title = summary
      ? [summary.config, summary.controller, summary.seed !== undefined ? `seed ${summary.seed}` : null]
          .filter(Boolean)
          .join(" · ")
      : name.replace(/\.npz$/, "");

    return {
      name,
      title: title || name,
      verdict: (summary && summary.verdict) || null,
      detail: (summary && summary.detail) || null,
      n: steps,
      t,
      pos,
      des,
      commandedHeight,
      controlMode,
      quat,
      speed,
      speedMin,
      speedMax,
      speedFraction: (value) => (span > 1e-9 ? (value - speedMin) / span : 0),
      dt,
      simTime: scalar(summary && summary.sim_time, t[steps - 1]),
      nFinal: scalar(summary && summary.n_final, passed[steps - 1]),
      nRequired: scalar(summary && summary.n_required, sequence.length),
      gates,
      poles,
      passes: buildPasses(passed, t, sequence),
      gateSequence: sequence,
      sensorRange,
      view,
      zMin: Math.min(0, zmin) - 0.1,
      zMax: zmax + 0.15,
    };
  }

  NS.model = { build, revealSteps };
})(typeof globalThis !== "undefined" ? globalThis : this);
