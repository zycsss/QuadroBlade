# replay_site — replay a debug run in 2D and 3D

The debug harness writes one `.npz` log per episode. The figure it draws is a still; this folder
turns the same log back into the flight: a top view, an orbitable 3D view, and altitude and speed
against time, all scrubbable with a slider. It can open a single log, a whole folder of them, or the
`logs/` directory of this repository.

Nothing is re-derived from the controller or the environment. Every pose, every gate and every pole
comes out of the log, so the replay can only show what was actually recorded.

## Run it

Three ways, none of them needing a build step or a network connection.

**With the picker** (browse every log in `logs/`):

```bash
pixi run python replay_site/serve.py
```

That prints a URL, opens it, and lists the run logs newest first. `?log=<file>` and `?step=<n>` in the
URL make a specific run and moment shareable. Pass `--log-dir` to browse another directory, `--port`
to change the port, and `--no-browser` to stay quiet.

**From the page itself** (no server at all): open `index.html` and press *Open folder*. The folder and
all its subfolders are scanned for `.npz` files, every one of them is added to the run picker, and the
newest is opened. *Open .npz* takes individual files, and dropping a folder (or a set of files) on the
page does the same. A log is paired with the `.json` summary of the same name when there is one, so the
header shows the verdict and the configuration.

The scan happens in the browser: the page reads the files the browser hands it and uploads nothing. It
is bounded (12 levels deep, 5000 candidate files) so picking a home directory by mistake cannot hang
the page. The served list and the scanned files share one picker, grouped by where they came from.

## What it shows

| | |
| --- | --- |
| **2D** | the track from above, same geometry and same speed ramp as the debug figure |
| **3D** | the same run in perspective, orbitable — gate openings are squares in space, poles are vertical capsules, and a drop line under the drone shows its height |
| **trajectory** | the flown path, every segment coloured by the speed at that moment (legend under the views) |
| **gates and poles** | both positions: the *nominal* one from the config the controller started with, and the *revealed* true one. Until the drone comes within the sensor range the object is drawn dashed and labelled `g0 ?`; after that the true pose is solid and the nominal one stays as a faint ghost with a line to where it really was |
| **sensor range** | the circle around the drone that decides when a reveal happens |
| **chart** | altitude and speed against time for the whole run, with every gate pass marked and the controlled height dashed where the controller commanded a position |

The reveal is read from the logged `gates_visited`/`poles_visited` flags — the same flags the
controller was given. A log written before those were recorded falls back to the environment's rule
(horizontal distance below `sensor_range`, permanently, once reached), so old logs still replay.

## Controls

| | |
| --- | --- |
| `Space` | play / pause |
| `←` `→` | step one control step |
| `Home` `End` | first / last step |
| slider, `◀` `▶` buttons | scrub and step |
| `rate` | playback speed, 0.25× to 2× of real time |
| click the chart | seek to that moment |
| drag the 3D view | orbit |
| wheel | zoom |
| double-click the 3D view | reset the camera |
| *reveal all* | ignore the reveal timeline and show every true pose immediately |

## The log it reads

Anything `scripts/debug_run.py` writes. The replay uses:

* `t`, `pos`, `quat`, `speed` — the flight;
* `gates_pos`/`gates_quat` and `obstacles_pos` — the nominal track;
* `true_gates_pos`/`true_gates_quat` and `true_obstacles_pos` — the randomized track;
* `gates_visited`, `poles_visited`, `sensor_range` — when a true pose became known;
* `n`, `gate_sequence` — gate passes, for the chart markers and the readout;
* `des` and `control_mode` — the dashed commanded height, drawn only in `state` control mode, where a
  position is actually commanded (in `attitude` mode the same field is roll/pitch/yaw);
* the sibling `.json` — title, verdict, timing.

`.npz` is a ZIP of `.npy` arrays, and both are read in the browser (`assets/npz.js`), including the
deflate payload from `savez_compressed`. That is why a log can be dropped onto a page opened with
`file://`.

## Files

```
index.html            the page
assets/style.css      the chrome around the canvases
assets/npz.js         ZIP + .npy reader
assets/scan.js        folder and drop scanning: walk, pair summaries, merge, label
assets/model.js       log arrays -> reveal timeline, bounds, passes
assets/util.js        geometry, speed ramp, quaternions, canvas helpers
assets/view2d.js      top view
assets/view3d.js      perspective view and its orbit camera
assets/chart.js       altitude and speed against time
assets/app.js         loading, playback clock, controls
serve.py              static server + /api/logs, /logs/<file>, /plots/<file>
```

The 3D view is line art drawn with the canvas 2D API and a painter's algorithm (project, sort by
depth, draw far to near) — no WebGL and no third-party renderer, so the folder stays self-contained.
