// serve/web/qr.js - a small QR code encoder (byte mode, error correction L or M, versions 1-10: up to 271 bytes at L),
// written for this project after the ISO/IEC 18004 standard, for the phone card's link.  It follows the structure of
// Project Nayuki's QR Code generator library (MIT License), whose notice is kept here:
//
//   Copyright (c) Project Nayuki. (MIT License)  https://www.nayuki.io/page/qr-code-generator-library
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
//   to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//   - The above copyright notice and this permission notice shall be included in all copies or substantial portions
//     of the Software.
//   - The Software is provided "as is", without warranty of any kind, express or implied, including but not limited
//     to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall
//     the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of
//     contract, tort or otherwise, arising from, out of or in connection with the Software or the use or other
//     dealings in the Software.
//
// qrMatrix(text, "M") -> array of rows of booleans (true = dark), or null when the text is too long.
// qrSvg(text) -> an SVG string (a quiet zone of 4 modules included).
"use strict";

const QR = (() => {
  const ECC = {L: {bits: 1, per: [0, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18], blocks: [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4]},
               M: {bits: 0, per: [0, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26], blocks: [0, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5]}};

  function rawModules(ver) {                       // modules left for data and error correction
    let r = (16 * ver + 128) * ver + 64;
    if (ver >= 2) {
      const n = Math.floor(ver / 7) + 2;
      r -= (25 * n - 10) * n - 55;
      if (ver >= 7) r -= 36;
    }
    return r;
  }
  const dataCodewords = (ver, e) => Math.floor(rawModules(ver) / 8) - e.per[ver] * e.blocks[ver];

  // Reed-Solomon over GF(256), polynomial 0x11D
  function mul(x, y) {
    let z = 0;
    for (let i = 7; i >= 0; i--) {
      z = (z << 1) ^ ((z >>> 7) * 0x11D);
      z ^= ((y >>> i) & 1) * x;
    }
    return z & 0xFF;
  }
  function divisor(degree) {
    const r = new Array(degree).fill(0);
    r[degree - 1] = 1;
    let root = 1;
    for (let i = 0; i < degree; i++) {
      for (let j = 0; j < r.length; j++) {
        r[j] = mul(r[j], root);
        if (j + 1 < r.length) r[j] ^= r[j + 1];
      }
      root = mul(root, 0x02);
    }
    return r;
  }
  function remainder(data, div) {
    const r = new Array(div.length).fill(0);
    for (const b of data) {
      const f = b ^ r.shift();
      r.push(0);
      div.forEach((c, i) => { r[i] ^= mul(c, f); });
    }
    return r;
  }

  // the data codewords: byte mode, the length, the bytes, terminator and padding
  function encodeData(bytes, ver, e) {
    const bits = [];
    const put = (v, n) => { for (let i = n - 1; i >= 0; i--) bits.push((v >>> i) & 1); };
    put(4, 4);
    put(bytes.length, ver <= 9 ? 8 : 16);
    bytes.forEach((b) => put(b, 8));
    const cap = dataCodewords(ver, e) * 8;
    put(0, Math.min(4, cap - bits.length));
    put(0, (8 - (bits.length % 8)) % 8);
    for (let p = 0xEC; bits.length < cap; p ^= 0xEC ^ 0x11) put(p, 8);
    const out = [];
    for (let i = 0; i < bits.length; i += 8) out.push(bits.slice(i, i + 8).reduce((a, b) => (a << 1) | b, 0));
    return out;
  }
  // split into blocks, add each block's error correction, interleave
  function withEcc(data, ver, e) {
    const nb = e.blocks[ver], ecLen = e.per[ver], raw = Math.floor(rawModules(ver) / 8);
    const short = nb - (raw % nb), shortLen = Math.floor(raw / nb), div = divisor(ecLen);
    const blocks = [];
    for (let i = 0, k = 0; i < nb; i++) {
      const dat = data.slice(k, k + shortLen - ecLen + (i < short ? 0 : 1));
      k += dat.length;
      const ecc = remainder(dat, div);
      if (i < short) dat.push(0);                  // a placeholder, skipped below
      blocks.push(dat.concat(ecc));
    }
    const out = [];
    for (let i = 0; i < blocks[0].length; i++) {
      blocks.forEach((b, j) => { if (i !== shortLen - ecLen || j >= short) out.push(b[i]); });
    }
    return out;
  }
  function alignPositions(ver) {
    if (ver === 1) return [];
    const n = Math.floor(ver / 7) + 2;
    const step = Math.ceil((ver * 4 + 4) / (n * 2 - 2)) * 2;
    const r = [6];
    for (let pos = ver * 4 + 10; r.length < n; pos -= step) r.splice(1, 0, pos);
    return r;
  }
  function formatBits(e, mask) {
    const data = (e.bits << 3) | mask;
    let rem = data;
    for (let i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >>> 9) * 0x537);
    return ((data << 10) | rem) ^ 0x5412;
  }
  function versionBits(ver) {
    let rem = ver;
    for (let i = 0; i < 12; i++) rem = (rem << 1) ^ ((rem >>> 11) * 0x1F25);
    return (ver << 12) | rem;
  }
  const MASKS = [
    (x, y) => (x + y) % 2 === 0, (x, y) => y % 2 === 0, (x) => x % 3 === 0, (x, y) => (x + y) % 3 === 0,
    (x, y) => (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0, (x, y) => (x * y) % 2 + (x * y) % 3 === 0,
    (x, y) => ((x * y) % 2 + (x * y) % 3) % 2 === 0, (x, y) => ((x + y) % 2 + (x * y) % 3) % 2 === 0,
  ];

  function build(ver, e, codewords, mask) {
    const size = ver * 4 + 17;
    const m = Array.from({length: size}, () => new Array(size).fill(false));
    const fn = Array.from({length: size}, () => new Array(size).fill(false));
    const set = (x, y, dark) => { m[y][x] = dark; fn[y][x] = true; };
    for (let i = 0; i < size; i++) { set(6, i, i % 2 === 0); set(i, 6, i % 2 === 0); }      // timing
    const finder = (cx, cy) => {
      for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
        const x = cx + dx, y = cy + dy, d = Math.max(Math.abs(dx), Math.abs(dy));
        if (x >= 0 && x < size && y >= 0 && y < size) set(x, y, d !== 2 && d !== 4);
      }
    };
    finder(3, 3); finder(size - 4, 3); finder(3, size - 4);
    const al = alignPositions(ver), n = al.length;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if ((i === 0 && j === 0) || (i === 0 && j === n - 1) || (i === n - 1 && j === 0)) continue;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        set(al[i] + dx, al[j] + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1);
      }
    }
    const drawFormat = (msk) => {
      const b = formatBits(e, msk), bit = (i) => ((b >>> i) & 1) !== 0;
      for (let i = 0; i <= 5; i++) set(8, i, bit(i));
      set(8, 7, bit(6)); set(8, 8, bit(7)); set(7, 8, bit(8));
      for (let i = 9; i < 15; i++) set(14 - i, 8, bit(i));
      for (let i = 0; i < 8; i++) set(size - 1 - i, 8, bit(i));
      for (let i = 8; i < 15; i++) set(8, size - 15 + i, bit(i));
      set(8, size - 8, true);                      // the dark module
    };
    drawFormat(0);                                 // reserves the area; drawn again with the real mask below
    if (ver >= 7) {
      const b = versionBits(ver);
      for (let i = 0; i < 18; i++) {
        const dark = ((b >>> i) & 1) !== 0, a = size - 11 + (i % 3), c = Math.floor(i / 3);
        set(a, c, dark); set(c, a, dark);
      }
    }
    let i = 0;
    for (let right = size - 1; right >= 1; right -= 2) {          // the zigzag, two columns at a time
      if (right === 6) right = 5;
      for (let v = 0; v < size; v++) for (let j = 0; j < 2; j++) {
        const x = right - j, up = ((right + 1) & 2) === 0, y = up ? size - 1 - v : v;
        if (!fn[y][x] && i < codewords.length * 8) {
          m[y][x] = ((codewords[i >>> 3] >>> (7 - (i & 7))) & 1) !== 0;
          i++;
        }
      }
    }
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (!fn[y][x] && MASKS[mask](x, y)) m[y][x] = !m[y][x];
    drawFormat(mask);
    return m;
  }

  function penalty(m) {                            // the standard's four rules (lower reads better)
    const size = m.length;
    let p = 0;
    const line = (get) => {
      for (let a = 0; a < size; a++) {
        let run = 1;
        for (let b = 1; b <= size; b++) {
          if (b < size && get(a, b) === get(a, b - 1)) run++;
          else { if (run >= 5) p += run - 2; run = 1; }
        }
        for (let b = 0; b + 7 <= size; b++) {      // 1:1:3:1:1 with four light modules on one side
          const s = [0, 1, 2, 3, 4, 5, 6].map((k) => get(a, b + k));
          if (s[0] && !s[1] && s[2] && s[3] && s[4] && !s[5] && s[6]) {
            const before = b >= 4 && [1, 2, 3, 4].every((k) => !get(a, b - k));
            const after = b + 11 <= size && [7, 8, 9, 10].every((k) => !get(a, b + k));
            if (before || after) p += 40;
          }
        }
      }
    };
    line((y, x) => m[y][x]);
    line((x, y) => m[y][x]);
    let dark = 0;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      if (m[y][x]) dark++;
      if (x + 1 < size && y + 1 < size && m[y][x] === m[y][x + 1] && m[y][x] === m[y + 1][x] && m[y][x] === m[y + 1][x + 1]) p += 3;
    }
    p += Math.floor(Math.abs(dark * 20 - size * size * 10) / (size * size)) * 10;
    return p;
  }

  function qrMatrix(text, level = "M") {
    const e = ECC[level] || ECC.M;
    const bytes = Array.from(new TextEncoder().encode(text));
    let ver = 1;
    while (ver <= 10 && 4 + (ver <= 9 ? 8 : 16) + bytes.length * 8 > dataCodewords(ver, e) * 8) ver++;
    if (ver > 10) return null;
    const cw = withEcc(encodeData(bytes, ver, e), ver, e);
    let best = null, bestP = Infinity;
    for (let mask = 0; mask < 8; mask++) {
      const m = build(ver, e, cw, mask), p = penalty(m);
      if (p < bestP) { best = m; bestP = p; }
    }
    return best;
  }
  function qrSvg(text, level = "M") {
    const m = qrMatrix(text, level);
    if (!m) return "";
    const n = m.length + 8;
    let d = "";
    m.forEach((row, y) => row.forEach((dark, x) => { if (dark) d += `M${x + 4},${y + 4}h1v1h-1z`; }));
    return `<svg class="qr" viewBox="0 0 ${n} ${n}" shape-rendering="crispEdges" role="img" aria-label="QR code">` +
           `<rect width="${n}" height="${n}" fill="#fff"/><path d="${d}" fill="#000"/></svg>`;
  }
  return {qrMatrix, qrSvg, _test: {mul, divisor, remainder, formatBits, versionBits, alignPositions, dataCodewords, ECC}};
})();
