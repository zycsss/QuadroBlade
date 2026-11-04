/**
 * Wiring: open a log, drive the clock, and keep the two views and the chart in step.
 *
 * Logs arrive from one of three places: the picker over the `logs/` directory that `serve.py` lists,
 * a folder picked from disk, or a folder (or a set of files) dropped on the page. The last two are
 * what make the page work from `file://`, and they are listed in the same picker as the served ones.
 * All three end in the same place: an array buffer handed to `Replay.npz.read`.
 */
(function (root) {
  "use strict";

  const NS = root.Replay;

  const state = {
    model: null,
    step: 0,
    playing: false,
    rate: 1,
    revealedAll: false,
    camera: null,
    carry: 0,
    lastTimestamp: 0,
    chartHit: null,
    runs: [],
    localRuns: [],
  };

  const el = {};

  /**
   * Look up an element by id.
   *
   * @param {string} id Element id.
   * @returns {HTMLElement} The element.
   */
  function byId(id) {
    return document.getElementById(id);
  }

  /**
   * Write the status line under the page.
   *
   * @param {string} text Message.
   * @param {string} [tone] `warn` or `error` to colour it.
   */
  function setStatus(text, tone) {
    el.status.textContent = text;
    el.status.className = "status" + (tone ? ` ${tone}` : "");
  }

  /**
   * Draw the three canvases from the current state.
   */
  function render() {
    if (!state.model) return;
    NS.view2d.draw(el.view2d, state.model, state);
    NS.view3d.draw(el.view3d, state.model, state);
    state.chartHit = NS.chart.draw(el.chart, state.model, state);
  }

  /**
   * One-line summary of the run for the header.
   *
   * @param {Object} model Replay model.
   * @returns {string} Text for the subtitle.
   */
  function describe(model) {
    const parts = [model.title];
    if (model.verdict) parts.push(model.verdict);
    parts.push(`${model.n} steps`, `${model.simTime.toFixed(2)} s`);
    if (model.gates.length) parts.push(`${model.gates.length} gates`);
    if (model.poles.length) parts.push(`${model.poles.length} poles`);
    return parts.join(" · ");
  }

  /**
   * Move the playback position and redraw.
   *
   * @param {number} value Step index.
   */
  function setStep(value) {
    if (!state.model || !Number.isFinite(value)) return;
    const step = Math.min(state.model.n - 1, Math.max(0, Math.round(value)));
    state.step = step;
    el.slider.value = String(step);
    updateReadout();
    render();
  }

  /**
   * Refresh the text beside the slider.
   */
  function updateReadout() {
    const model = state.model;
    const step = state.step;
    const point = model.pos[step];
    const passes = model.passes.filter((pass) => pass.step <= step).length;
    el.readout.textContent =
      `t ${model.t[step].toFixed(2)} s · ${step + 1}/${model.n} · ` +
      `z ${point[2].toFixed(2)} m · speed ${model.speed[step].toFixed(2)} m/s · ` +
      `${passes}/${model.nRequired} gates`;
  }

  /**
   * Adopt a newly loaded model.
   *
   * @param {Object} model Replay model from `Replay.model.build`.
   */
  function showModel(model) {
    state.model = model;
    state.playing = false;
    state.carry = 0;
    state.lastTimestamp = 0;
    state.step = model.n - 1;
    state.camera = NS.view3d.defaultCamera(model);

    el.play.textContent = "Play";
    el.slider.min = "0";
    el.slider.max = String(model.n - 1);
    el.slider.value = String(state.step);
    el.subtitle.textContent = describe(model);
    el.rampMin.textContent = model.speedMin.toFixed(2);
    el.rampMax.textContent = model.speedMax.toFixed(2);
    el.rampBar.style.background = NS.util.speedGradientCss();
    el.reveal.checked = state.revealedAll;
    document.title = `QuadroBlade · ${model.name.replace(/\.npz$/, "")}`;
    updateReadout();
    render();
  }

  /**
   * Parse a log buffer and show it.
   *
   * @param {string} name File name, used as the run title fallback.
   * @param {ArrayBuffer} buffer Contents of the `.npz`.
   * @param {Object|null} summary Parsed `.json` summary, or null.
   * @returns {Promise<Object>} The model that was shown.
   */
  async function loadFromBuffer(name, buffer, summary) {
    const parsed = await NS.npz.read(buffer);
    const model = NS.model.build(parsed.arrays, summary, name);
    showModel(model);
    setStatus(
      `loaded ${name} · ${parsed.names.length} arrays · ${model.n} steps` +
        (model.verdict ? ` · ${model.verdict}` : "")
    );
    return model;
  }

  /**
   * Load one run that the server listed.
   *
   * @param {{name: string, summary: Object|null, plot: boolean}} run Entry from `/api/logs`.
   * @returns {Promise<Object|null>} The model that was shown, or null.
   */
  async function loadServerRun(run) {
    if (!run) return null;
    try {
      setStatus(`loading ${run.name}…`);
      const response = await fetch(`/logs/${encodeURIComponent(run.name)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status} for ${run.name}`);
      const buffer = await response.arrayBuffer();
      const model = await loadFromBuffer(run.name, buffer, run.summary || null);
      if (run.plot) {
        el.plotlink.href = `/plots/${encodeURIComponent(run.name.replace(/\.npz$/i, ".png"))}`;
        el.plotlink.hidden = false;
      } else {
        el.plotlink.hidden = true;
      }
      return model;
    } catch (error) {
      setStatus(`could not load ${run.name}: ${error.message}`, "error");
      return null;
    }
  }

  /**
   * Load one run that was scanned from disk. The file is already in hand, so no request is made.
   *
   * @param {{path: string, file: File, summary: Object|null}} run Entry from `Replay.scan.buildRuns`.
   * @returns {Promise<Object|null>} The model that was shown, or null.
   */
  async function loadLocalRun(run) {
    if (!run) return null;
    try {
      setStatus(`reading ${run.path}…`);
      const buffer = await run.file.arrayBuffer();
      const model = await loadFromBuffer(run.file.name, buffer, run.summary || null);
      el.plotlink.hidden = true;
      return model;
    } catch (error) {
      setStatus(`could not read ${run.path}: ${error.message}`, "error");
      return null;
    }
  }

  /**
   * Load whatever the picker currently points at.
   *
   * @param {string} value Option value: `srv:<index>` or `loc:<index>`.
   * @returns {Promise<Object|null>} The model that was shown, or null.
   */
  async function selectRun(value) {
    if (!value) return null;
    if (value.startsWith("srv:")) return loadServerRun(state.runs[Number(value.slice(4))]);
    if (value.startsWith("loc:")) return loadLocalRun(state.localRuns[Number(value.slice(4))]);
    return null;
  }

  /**
   * Fill the run picker: the served `logs/` directory first, then everything scanned from disk.
   */
  function fillRuns() {
    el.runs.innerHTML = "";
    if (state.runs.length) {
      const group = document.createElement("optgroup");
      group.label = `logs/ (${state.runs.length})`;
      state.runs.forEach((run, index) => {
        const option = document.createElement("option");
        option.value = `srv:${index}`;
        const verdict = run.summary && run.summary.verdict ? ` · ${run.summary.verdict}` : "";
        option.textContent = run.name.replace(/\.npz$/i, "") + verdict;
        group.appendChild(option);
      });
      el.runs.appendChild(group);
    }
    if (state.localRuns.length) {
      const labels = NS.scan.labelsFor(state.localRuns);
      const group = document.createElement("optgroup");
      group.label = `scanned from disk (${state.localRuns.length})`;
      state.localRuns.forEach((run, index) => {
        const option = document.createElement("option");
        option.value = `loc:${index}`;
        option.title = run.path;
        const verdict = run.summary && run.summary.verdict ? ` · ${run.summary.verdict}` : "";
        option.textContent = labels[index].replace(/\.npz$/i, "") + verdict;
        group.appendChild(option);
      });
      el.runs.appendChild(group);
    }
    el.runs.hidden = !(state.runs.length || state.localRuns.length);
  }

  /**
   * On a served page, list the logs in the sibling `logs/` directory and open the newest one.
   */
  async function autoload() {
    try {
      const response = await fetch("/api/logs", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      state.runs = data.runs || [];
      fillRuns();
      if (!state.runs.length) {
        setStatus(
          "no run logs in logs/ yet — scan a folder, or run scripts/debug_run.py and reload", "warn"
        );
        return;
      }
      const wanted = new URLSearchParams(root.location.search).get("log");
      const run = state.runs.find((entry) => entry.name === wanted) || state.runs[0];
      el.runs.value = `srv:${state.runs.indexOf(run)}`;
      const model = await loadServerRun(run);
      // `?step=` makes a moment shareable, e.g. the gate a run failed at.
      const wantedStep = new URLSearchParams(root.location.search).get("step");
      if (model && wantedStep !== null) setStep(Number(wantedStep));
    } catch (error) {
      setStatus(
        "open a folder or drop a run log, or run replay_site/serve.py to browse logs/ from here"
      );
    }
  }

  /**
   * How many steps to advance for the time that just passed.
   *
   * @param {number} timestamp `requestAnimationFrame` timestamp in milliseconds.
   */
  function frame(timestamp) {
    root.requestAnimationFrame(frame);
    if (!state.model || !state.playing) {
      state.lastTimestamp = timestamp;
      return;
    }
    if (!state.lastTimestamp) state.lastTimestamp = timestamp;
    const realSeconds = Math.min(0.1, (timestamp - state.lastTimestamp) / 1000);
    state.lastTimestamp = timestamp;
    state.carry += realSeconds * state.rate;
    const stepSeconds = Math.max(0.001, state.model.dt);
    let advance = Math.floor(state.carry / stepSeconds);
    if (advance <= 0) return;
    state.carry -= advance * stepSeconds;
    advance = Math.min(advance, 20);
    if (state.step + advance >= state.model.n - 1) {
      setStep(state.model.n - 1);
      pause();
    } else {
      setStep(state.step + advance);
    }
  }

  /** Start or resume playback, rewinding first when the run is already at its end. */
  function play() {
    if (!state.model) return;
    if (state.step >= state.model.n - 1) setStep(0);
    state.playing = true;
    state.lastTimestamp = 0;
    state.carry = 0;
    el.play.textContent = "Pause";
  }

  /** Stop playback. */
  function pause() {
    state.playing = false;
    state.lastTimestamp = 0;
    el.play.textContent = "Play";
  }

  /**
   * Adopt the run logs found by a scan: add them to the picker and show the newest.
   *
   * @param {Array<{path: string, file: File}>} scanned Candidate files from `Replay.scan`.
   * @param {string|null} source Folder the files came from, or null for loose files.
   * @returns {Promise<void>} Resolves once the newest is shown.
   */
  async function adoptScanned(scanned, source) {
    const where = source ? ` in ${source}` : "";
    if (!scanned.length) {
      setStatus(`no .npz run logs found${where}`, "warn");
      return;
    }
    const runs = await NS.scan.buildRuns(scanned);
    if (!runs.length) {
      setStatus(`no .npz run logs found${where}`, "warn");
      return;
    }
    state.localRuns = NS.scan.merge(state.localRuns, runs);
    fillRuns();
    const newest = runs[0];
    const value = `loc:${state.localRuns.indexOf(newest)}`;
    el.runs.value = value;
    const model = await selectRun(value);
    const plural = runs.length === 1 ? "log" : "logs";
    setStatus(
      `scanned ${runs.length} ${plural}${where}` +
        (model ? ` · showing ${newest.name} (${model.n} steps)` : "")
    );
  }

  /**
   * Scan what a file or directory input selected, then clear it so the same choice can be repeated.
   *
   * @param {HTMLInputElement} input The input element.
   * @returns {Promise<void>} Resolves once the newest log is shown.
   */
  async function scanInput(input) {
    const scanned = NS.scan.fromInput(input.files);
    input.value = "";
    await adoptScanned(scanned, NS.scan.sourceLabel(scanned));
  }

  /**
   * Scan whatever was dropped: one or more files, or one or more folders.
   *
   * @param {DataTransfer} dataTransfer Drop payload.
   * @returns {Promise<void>} Resolves once the newest log is shown.
   */
  async function scanDrop(dataTransfer) {
    const scanned = await NS.scan.fromDataTransfer(dataTransfer);
    await adoptScanned(scanned, NS.scan.sourceLabel(scanned));
  }

  /**
   * Attach every listener the page needs.
   */
  function wireControls() {
    el.play.addEventListener("click", () => (state.playing ? pause() : play()));
    el.back.addEventListener("click", () => {
      pause();
      setStep(state.step - 1);
    });
    el.forward.addEventListener("click", () => {
      pause();
      setStep(state.step + 1);
    });
    el.slider.addEventListener("input", () => {
      pause();
      setStep(Number(el.slider.value));
    });
    el.reveal.addEventListener("change", () => {
      state.revealedAll = el.reveal.checked;
      render();
    });
    el.rate.addEventListener("change", () => {
      state.rate = Number(el.rate.value) || 1;
    });
    el.file.addEventListener("change", () => scanInput(el.file));
    el.folder.addEventListener("change", () => scanInput(el.folder));
    el.runs.addEventListener("change", () => selectRun(el.runs.value));

    NS.view3d.attachControls(el.view3d, state, render);

    el.chart.addEventListener("click", (event) => {
      if (!state.model || !state.chartHit) return;
      const rect = el.chart.getBoundingClientRect();
      pause();
      setStep(state.chartHit.stepAt(event.clientX - rect.left));
    });

    root.addEventListener("resize", render);

    let depth = 0;
    root.addEventListener("dragenter", (event) => {
      if (!event.dataTransfer || !Array.from(event.dataTransfer.types).includes("Files")) return;
      depth++;
      el.drop.hidden = false;
    });
    root.addEventListener("dragleave", () => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) el.drop.hidden = true;
    });
    root.addEventListener("dragover", (event) => event.preventDefault());
    root.addEventListener("drop", (event) => {
      event.preventDefault();
      depth = 0;
      el.drop.hidden = true;
      // `scanDrop` reads the entries synchronously, before the drop payload goes stale.
      scanDrop(event.dataTransfer);
    });

    root.addEventListener("keydown", (event) => {
      const tag = (event.target && event.target.tagName) || "";
      if (/INPUT|SELECT|TEXTAREA/.test(tag)) return;
      if (event.code === "Space") {
        event.preventDefault();
        if (state.playing) pause();
        else play();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        pause();
        setStep(state.step - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        pause();
        setStep(state.step + 1);
      } else if (event.key === "Home") {
        pause();
        setStep(0);
      } else if (event.key === "End") {
        pause();
        setStep(state.model ? state.model.n - 1 : 0);
      }
    });
  }

  /** Look up the page elements, wire them, and start the clock. */
  function boot() {
    el.view2d = byId("view2d");
    el.view3d = byId("view3d");
    el.chart = byId("chart");
    el.slider = byId("slider");
    el.play = byId("play");
    el.back = byId("back");
    el.forward = byId("forward");
    el.reveal = byId("reveal");
    el.rate = byId("rate");
    el.file = byId("file");
    el.folder = byId("folder");
    el.runs = byId("runs");
    el.plotlink = byId("plotlink");
    el.readout = byId("readout");
    el.subtitle = byId("subtitle");
    el.rampMin = byId("rampMin");
    el.rampMax = byId("rampMax");
    el.rampBar = byId("rampBar");
    el.status = byId("status");
    el.drop = byId("drop");

    wireControls();
    root.requestAnimationFrame(frame);
    autoload();
  }

  NS.app = { state, loadFromBuffer };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})(typeof globalThis !== "undefined" ? globalThis : this);
