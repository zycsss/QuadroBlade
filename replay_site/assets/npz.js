/**
 * Reader for the `.npz` run logs that `scripts/debug_run.py` writes.
 *
 * An `.npz` is a ZIP archive with one `.npy` file per array, and a `.npy` is a short text header
 * followed by one flat binary buffer. Both formats are simple enough to read here, which means a log
 * can be dropped straight into the page: no server, no build step, no Python, and no chance of the
 * replay showing something other than what was logged.
 *
 * The reader returns plain nested JavaScript arrays (e.g. `pos[i][2]`), because the arrays in a run
 * log are small - a few hundred steps at most - and readable indexing is worth more here than the
 * copy that a typed-array-per-stride design would avoid.
 *
 * Exposed as `Replay.npz.read(arrayBuffer)`.
 */
(function (root) {
  "use strict";

  const NS = (root.Replay = root.Replay || {});

  const EOCD_SIGNATURE = 0x06054b50;
  const CENTRAL_SIGNATURE = 0x02014b50;
  const LOCAL_SIGNATURE = 0x04034b50;

  /**
   * Find the End Of Central Directory record, which is the last 22+ bytes of the archive.
   *
   * @param {DataView} view Whole-archive view.
   * @returns {number} Byte offset of the record, or -1.
   */
  function findCentralDirectoryEnd(view) {
    // The record is followed only by a comment of at most 65535 bytes, so it cannot be further
    // than that from the end.
    const earliest = Math.max(0, view.byteLength - (0xffff + 22));
    for (let offset = view.byteLength - 22; offset >= earliest; offset--) {
      if (view.getUint32(offset, true) === EOCD_SIGNATURE) return offset;
    }
    return -1;
  }

  /**
   * List the archive entries with the few fields needed to extract them.
   *
   * @param {DataView} view Whole-archive view.
   * @returns {Array<{name: string, method: number, compressedSize: number, offset: number}>}
   */
  function readDirectory(view) {
    const end = findCentralDirectoryEnd(view);
    if (end < 0) throw new Error("not a .npz file: no ZIP central directory found");
    const count = view.getUint16(end + 10, true);
    const decoder = new TextDecoder("utf-8");
    const entries = [];
    let cursor = view.getUint32(end + 16, true);
    for (let i = 0; i < count; i++) {
      if (view.getUint32(cursor, true) !== CENTRAL_SIGNATURE) {
        throw new Error("corrupt .npz: bad central directory entry");
      }
      const method = view.getUint16(cursor + 10, true);
      const compressedSize = view.getUint32(cursor + 20, true);
      const nameLength = view.getUint16(cursor + 28, true);
      const extraLength = view.getUint16(cursor + 30, true);
      const commentLength = view.getUint16(cursor + 32, true);
      const localOffset = view.getUint32(cursor + 42, true);
      const nameBytes = new Uint8Array(
        view.buffer,
        view.byteOffset + cursor + 46,
        nameLength
      );
      entries.push({
        name: decoder.decode(nameBytes),
        method,
        compressedSize,
        offset: localOffset,
      });
      cursor += 46 + nameLength + extraLength + commentLength;
    }
    return entries;
  }

  /**
   * Raw (still compressed) bytes of one entry, taken from its local header.
   *
   * @param {DataView} view Whole-archive view.
   * @param {{method: number, compressedSize: number, offset: number}} entry Entry from `readDirectory`.
   * @returns {Uint8Array} The stored or deflated payload.
   */
  function entryBytes(view, entry) {
    if (view.getUint32(entry.offset, true) !== LOCAL_SIGNATURE) {
      throw new Error(`corrupt .npz: bad local header for ${entry.name}`);
    }
    const nameLength = view.getUint16(entry.offset + 26, true);
    const extraLength = view.getUint16(entry.offset + 28, true);
    const start = entry.offset + 30 + nameLength + extraLength;
    return new Uint8Array(view.buffer, view.byteOffset + start, entry.compressedSize);
  }

  /**
   * Inflate a raw deflate stream with the platform's own decompressor.
   *
   * @param {Uint8Array} bytes Deflated bytes.
   * @returns {Promise<Uint8Array>} Inflated bytes.
   */
  async function inflateRaw(bytes) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error(
        "this browser cannot inflate .npz archives (no DecompressionStream); " +
          "use a current Chrome, Edge, Firefox or Safari"
      );
    }
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  /**
   * Byte strides of the dimensions of an array, for either memory order.
   *
   * @param {number[]} shape Dimension lengths.
   * @param {boolean} fortran Whether the buffer is column-major.
   * @param {number} elementSize Bytes per element.
   * @returns {number[]} Byte stride per dimension.
   */
  function stridesFor(shape, fortran, elementSize) {
    const strides = new Array(shape.length).fill(elementSize);
    if (fortran) {
      for (let i = 1; i < shape.length; i++) strides[i] = strides[i - 1] * shape[i - 1];
    } else {
      for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  /**
   * Turn a row-major element index into a byte offset through the strides.
   *
   * @param {number} index Row-major index.
   * @param {number[]} shape Dimension lengths.
   * @param {number[]} strides Byte stride per dimension.
   * @returns {number} Byte offset of the element.
   */
  function byteOffset(index, shape, strides) {
    let rest = index;
    let offset = 0;
    for (let d = shape.length - 1; d >= 0; d--) {
      const size = shape[d] || 1;
      const coordinate = rest % size;
      rest = (rest - coordinate) / size;
      offset += coordinate * strides[d];
    }
    return offset;
  }

  /**
   * Read the elements of a numpy dtype from the buffer.
   *
   * @param {DataView} view View of the whole `.npy` file.
   * @param {number} base Byte offset of the first element.
   * @param {string} descr Numpy dtype string, e.g. `'<f8'`, `'|b1'` or `'<U120'`.
   * @returns {{elementSize: number, read: (offset: number) => number|string}} Element reader.
   */
  function readerFor(view, base, descr) {
    const kind = descr[1];
    const width = Number(descr.slice(2));
    const little = descr[0] !== ">";
    const at = base;
    switch (kind) {
      case "f":
        return {
          elementSize: width,
          read: (o) =>
            width === 8 ? view.getFloat64(at + o, little) : view.getFloat32(at + o, little),
        };
      case "i":
        if (width === 8) return { elementSize: 8, read: (o) => Number(view.getBigInt64(at + o, little)) };
        if (width === 4) return { elementSize: 4, read: (o) => view.getInt32(at + o, little) };
        if (width === 2) return { elementSize: 2, read: (o) => view.getInt16(at + o, little) };
        return { elementSize: 1, read: (o) => view.getInt8(at + o) };
      case "u":
        if (width === 8) return { elementSize: 8, read: (o) => Number(view.getBigUint64(at + o, little)) };
        if (width === 4) return { elementSize: 4, read: (o) => view.getUint32(at + o, little) };
        if (width === 2) return { elementSize: 2, read: (o) => view.getUint16(at + o, little) };
        return { elementSize: 1, read: (o) => view.getUint8(at + o) };
      case "b":
        // Booleans are stored as one byte; keep them as 0/1 so they compare like numbers.
        return { elementSize: 1, read: (o) => view.getUint8(at + o) };
      case "U": {
        // numpy stores unicode as fixed-width little-endian UCS-4, padded with zeros. Decoding the
        // code units by hand avoids depending on a utf-32 decoder being compiled into the runtime.
        const elementSize = width * 4;
        return {
          elementSize,
          read: (o) => {
            let text = "";
            for (let i = 0; i < width; i++) {
              const code = view.getUint32(at + o + i * 4, little);
              if (code === 0) break;
              text += String.fromCharCode(code);
            }
            return text;
          },
        };
      }
      case "S": {
        return {
          elementSize: width,
          read: (o) => {
            let text = "";
            for (let i = 0; i < width; i++) {
              const code = view.getUint8(at + o + i);
              if (code === 0) break;
              text += String.fromCharCode(code);
            }
            return text;
          },
        };
      }
      default:
        throw new Error(`unsupported numpy dtype '${descr}' in the run log`);
    }
  }

  /**
   * Nest a flat, row-major list of values into arrays of the given shape.
   *
   * @param {Array<number|string>} values Row-major values.
   * @param {number[]} shape Dimension lengths.
   * @returns {*} Nested array, or the single value of a 0-dimensional array.
   */
  function nest(values, shape) {
    if (shape.length === 0) return values[0];
    let cursor = 0;
    const build = (dim) => {
      const out = new Array(shape[dim]);
      for (let i = 0; i < shape[dim]; i++) {
        out[i] = dim === shape.length - 1 ? values[cursor++] : build(dim + 1);
      }
      return out;
    };
    return build(0);
  }

  /**
   * Parse one `.npy` file.
   *
   * @param {Uint8Array} bytes File contents.
   * @returns {{data: *, shape: number[], descr: string}} The array.
   */
  function readNpy(bytes) {
    if (bytes.byteLength < 10) throw new Error("not a .npy file: too short");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const isNpy =
      view.getUint8(0) === 0x93 &&
      view.getUint8(1) === 0x4e && // N
      view.getUint8(2) === 0x55 && // U
      view.getUint8(3) === 0x4d && // M
      view.getUint8(4) === 0x50 && // P
      view.getUint8(5) === 0x59; // Y
    if (!isNpy) throw new Error("not a .npy file: bad magic");
    const major = view.getUint8(6);
    const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
    const headerStart = major === 1 ? 10 : 12;
    const headerText = new TextDecoder("utf-8").decode(
      bytes.subarray(headerStart, headerStart + headerLength)
    );
    const descr = /'descr':\s*'([^']+)'/.exec(headerText);
    const shapeText = /'shape':\s*\(([^)]*)\)/.exec(headerText);
    if (!descr || !shapeText) throw new Error("not a .npy file: unreadable header");
    const dtype = descr[1];
    const fortran = /'fortran_order':\s*True/.test(headerText);
    const shape = shapeText[1]
      .split(",")
      .map((part) => part.trim())
      .filter((part) => part.length > 0)
      .map(Number);
    const total = shape.reduce((product, size) => product * size, 1);
    const reader = readerFor(view, headerStart + headerLength, dtype);
    const strides = stridesFor(shape, fortran, reader.elementSize);
    const values = new Array(total);
    for (let i = 0; i < total; i++) values[i] = reader.read(byteOffset(i, shape, strides));
    return { data: nest(values, shape), shape, descr: dtype };
  }

  /**
   * Read a whole `.npz` archive.
   *
   * @param {ArrayBuffer} buffer File contents.
   * @returns {Promise<{arrays: Object<string, *>, names: string[]}>} The arrays by name, and the
   *     names in archive order.
   */
  async function read(buffer) {
    const view = new DataView(buffer);
    const entries = readDirectory(view);
    const arrays = {};
    const names = [];
    for (const entry of entries) {
      if (!entry.name.endsWith(".npy")) continue;
      const name = entry.name.replace(/\.npy$/, "").replace(/^.*\//, "");
      const stored = entryBytes(view, entry);
      const payload = entry.method === 0 ? stored : await inflateRaw(stored);
      arrays[name] = readNpy(payload).data;
      names.push(name);
    }
    if (!names.length) throw new Error("this .npz holds no arrays");
    return { arrays, names };
  }

  NS.npz = { read, readNpy };
})(typeof globalThis !== "undefined" ? globalThis : this);
