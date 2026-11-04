/**
 * Finding run logs: in a folder picked from disk, in a folder dropped on the page, or in a handful of
 * selected files.
 *
 * This is what makes the site usable without a server. A page opened with `file://` cannot fetch
 * anything, but it can read the files the user hands it, and a directory can be handed over in two
 * ways the browsers share: `<input webkitdirectory>` and `DataTransferItem.webkitGetAsEntry()`. Both
 * are walked here, so one folder pick and one folder drop end in the same list.
 *
 * Only `.npz` run logs and the `.json` summaries next to them are collected, so dropping a directory
 * that contains other things costs nothing.
 *
 * Exposed as `Replay.scan`.
 */
(function (root) {
  "use strict";

  const NS = (root.Replay = root.Replay || {});

  const LOG_SUFFIX = /\.npz$/i;
  const SUMMARY_SUFFIX = /\.json$/i;

  //: Refuse to walk pathological trees: a symlink loop or a home directory picked by mistake.
  const MAX_DEPTH = 12;
  const MAX_FILES = 5000;

  /**
   * Whether a file is worth keeping during the walk.
   *
   * @param {string} name File name.
   * @returns {boolean} True for a run log or a summary.
   */
  function interesting(name) {
    return LOG_SUFFIX.test(name) || SUMMARY_SUFFIX.test(name);
  }

  /**
   * Collect the run logs under one dropped entry, recursing into directories.
   *
   * @param {FileSystemEntry} entry File or directory entry from a drop.
   * @param {Array<{path: string, file: File}>} out Collected files, appended in place.
   * @param {string} prefix Path prefix of this entry's parent, ending in `/`.
   * @param {number} depth Recursion depth.
   * @returns {Promise<void>} Resolves when the subtree has been read.
   */
  function collectEntry(entry, out, prefix, depth) {
    if (!entry || depth > MAX_DEPTH || out.length >= MAX_FILES) return Promise.resolve();
    if (entry.isFile) {
      return new Promise((resolve) => {
        entry.file(
          (file) => {
            if (interesting(file.name) && out.length < MAX_FILES) {
              out.push({ path: `${prefix}${file.name}`, file });
            }
            resolve();
          },
          () => resolve()
        );
      });
    }
    if (!entry.isDirectory) return Promise.resolve();
    const reader = entry.createReader();
    return new Promise((resolve) => {
      // readEntries returns at most a few hundred entries per call, so it has to be called until it
      // comes back empty - one call would silently skip the rest of a large directory.
      const readNext = () => {
        reader.readEntries(
          (batch) => {
            if (!batch.length) {
              resolve();
              return;
            }
            Promise.all(
              batch.map((child) => collectEntry(child, out, `${prefix}${entry.name}/`, depth + 1))
            ).then(readNext, readNext);
          },
          () => resolve()
        );
      };
      readNext();
    });
  }

  /**
   * Read the entries of a drop. Handles dropped folders, dropped files, and browsers that expose no
   * entry API at all.
   *
   * The `webkitGetAsEntry` calls have to happen while the drop event is still being handled, which is
   * why they come before the first `await` in this function.
   *
   * @param {DataTransfer} dataTransfer Drop payload.
   * @returns {Promise<Array<{path: string, file: File}>>} The candidate files.
   */
  async function fromDataTransfer(dataTransfer) {
    const items = Array.from(dataTransfer.items || []);
    const entries = items
      .map((item) => {
        if (typeof item.webkitGetAsEntry !== "function") return null;
        try {
          return item.webkitGetAsEntry();
        } catch (error) {
          // Not every DataTransfer has real entries (a synthetic drop, for instance); the plain
          // file list below is the fallback.
          return null;
        }
      })
      .filter(Boolean);
    const out = [];
    if (entries.length) {
      for (const entry of entries) await collectEntry(entry, out, "", 0);
      return out;
    }
    for (const file of Array.from(dataTransfer.files || [])) {
      if (interesting(file.name)) out.push({ path: file.name, file });
    }
    return out;
  }

  /**
   * Read the files of an `<input type=file>` or `<input webkitdirectory>`.
   *
   * @param {FileList|File[]} fileList Selected files.
   * @returns {Array<{path: string, file: File}>} The candidate files, with the relative path when the
   *     browser supplies one.
   */
  function fromInput(fileList) {
    const out = [];
    for (const file of Array.from(fileList || [])) {
      if (!interesting(file.name)) continue;
      out.push({ path: file.webkitRelativePath || file.name, file });
    }
    return out;
  }

  /**
   * Order two runs: newest first, then by path so the order is stable.
   *
   * @param {{mtime: number, path: string}} a Run.
   * @param {{mtime: number, path: string}} b Run.
   * @returns {number} Sort order.
   */
  function compareRuns(a, b) {
    return (b.mtime || 0) - (a.mtime || 0) || a.path.localeCompare(b.path);
  }

  /**
   * Pair each run log with its summary and describe it.
   *
   * @param {Array<{path: string, file: File}>} scanned Files from `fromInput` or `fromDataTransfer`.
   * @returns {Promise<Array<Object>>} Runs, newest first, each with `path`, `name`, `size`, `mtime`,
   *     the parsed `summary` (or null) and the `file` to read.
   */
  async function buildRuns(scanned) {
    const summaries = new Map();
    const logs = new Map();
    for (const item of scanned) {
      if (SUMMARY_SUFFIX.test(item.path)) {
        summaries.set(item.path.replace(SUMMARY_SUFFIX, ""), item.file);
      } else if (LOG_SUFFIX.test(item.path)) {
        logs.set(item.path, item.file);
      }
    }
    const runs = [];
    for (const [path, file] of logs) {
      let summary = null;
      const summaryFile = summaries.get(path.replace(LOG_SUFFIX, ""));
      if (summaryFile) {
        try {
          summary = JSON.parse(await summaryFile.text());
        } catch (error) {
          summary = null;
        }
      }
      runs.push({
        path,
        name: file.name,
        size: file.size,
        mtime: file.lastModified || 0,
        summary,
        file,
      });
    }
    return runs.sort(compareRuns);
  }

  /**
   * Merge freshly scanned runs into an existing list, replacing an entry the user re-scanned.
   *
   * @param {Array<Object>} existing Runs already known.
   * @param {Array<Object>} incoming Runs just found.
   * @returns {Array<Object>} The merged list, newest first.
   */
  function merge(existing, incoming) {
    const byPath = new Map((existing || []).map((run) => [run.path, run]));
    for (const run of incoming || []) byPath.set(run.path, run);
    return Array.from(byPath.values()).sort(compareRuns);
  }

  /**
   * Display names for a list of runs: the file name when it is unique, the path when it is not.
   *
   * @param {Array<Object>} runs Runs to label.
   * @returns {string[]} One label per run.
   */
  function labelsFor(runs) {
    const counts = new Map();
    for (const run of runs) counts.set(run.name, (counts.get(run.name) || 0) + 1);
    return runs.map((run) => (counts.get(run.name) > 1 ? run.path : run.name));
  }

  /**
   * Where a scan came from, when it was one folder.
   *
   * @param {Array<{path: string}>} scanned The files that were found.
   * @returns {string|null} E.g. `folder "car-logs"`, or null when the files did not come from one
   *     folder (a handful of dropped or selected files).
   */
  function sourceLabel(scanned) {
    if (!scanned.length) return null;
    const roots = new Set(scanned.map((item) => item.path.split("/")[0]));
    const nested = scanned.every((item) => item.path.includes("/"));
    if (roots.size === 1 && nested) return `folder "${[...roots][0]}"`;
    return null;
  }

  NS.scan = {
    fromInput,
    fromDataTransfer,
    buildRuns,
    merge,
    labelsFor,
    sourceLabel,
    collectEntry,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
