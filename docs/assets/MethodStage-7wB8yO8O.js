// "How CoSeP chooses what to keep": the ten-step walkthrough of the method section.
// Hand-written ES module (no build step needed). React, the step player, the in-view and reduced-motion hooks
// and the illustration data come from the main bundle. Equations are pre-rendered with KaTeX 0.18.1 (the version
// this site already ships, whose stylesheet and fonts sit next to this file) and stored in the EQUATIONS block below.
import {
  a as ease,
  d as requireReact,
  f as toESM,
  i as lerp,
  n as DATA,
  o as useInView,
  r as gaussPdf,
  s as useReducedMotion,
  u as asset,
} from "./index-Itw68ETy.js";

const React = toESM(requireReact(), 1);
const { createElement: h, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } = React;

// ---------- palette (text >= 4.5:1, graphics >= 3:1 on white and on the plot panel) ----------
const C = {
  ink: "#0A0C0F", text: "#2B3038", muted: "#5A6472", faint: "#67707F",
  axis: "#7E8794", line: "#AEB5C0", grid: "#E3E6EA", panel: "#F6F7F9", panelEdge: "#E3E6EA",
  accent: "#3E63DD", accentText: "#2B47B0", accentTint: "#E8EDFC",
  knee: "#C25E00", kneeText: "#A34A04",
  keep: "#0D9488", keepText: "#0F766E", keepTint: "#DDF3F0",
  cut: "#DC3E42", node: "#9AA3B2",
};
const CLS = [
  { main: "#E5484D", text: "#BE1E2D", tint: "#FBE3E4" }, // bird
  { main: "#2B9A66", text: "#146C44", tint: "#DDF1E6" }, // automobile
  { main: "#B47D00", text: "#865A00", tint: "#F8ECCB" }, // horse
];
const GROUP = ["#3E63DD", "#E5484D", "#2B9A66", "#B47D00", "#8E4EC6", "#0D9488", "#D6409F", "#C25E00",
  "#0B7FC7", "#6E56CF", "#AD7F58", "#E54666", "#3E9B4F", "#978365", "#5B5BD6", "#12A594"];
const groupColor = (i) => GROUP[i % GROUP.length];

const SANS = "'Space Grotesk', ui-sans-serif, system-ui, sans-serif";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace";
const KMAIN = "KaTeX_Main, 'Times New Roman', serif";
const KMATH = "KaTeX_Math, 'Times New Roman', serif";
const KCAL = "KaTeX_Caligraphic, 'Times New Roman', serif";

// ---------- layout ----------
const VB = { wide: { w: 1000, h: 420 }, narrow: { w: 420, h: 600 } };
const NARROW_BELOW = 740; // card width (px) under which the portrait layout is used
const NARROW_MAX_FIG = 480; // px

// ---------- steps ----------
const ACTS = [
  { id: "inputs", label: "Inputs", title: "A small labeled sample", dwell: 9,
    caption: "CoSeP starts from a trained network and a labeled calibration set: three CIFAR-10 classes here, shown at their native 32 × 32 pixels. While it decides what to keep, labels serve one purpose only: to see how each component responds to each class." },
  { id: "forward", label: "Forward", title: "Through the network, to one layer", dwell: 14,
    caption: "The sample sorts into one stack per class, and every image goes through the same trained network. One layer is picked out, and everything that follows happens inside it." },
  { id: "activations", label: "Activations", title: "One number per neuron, per image", dwell: 15,
    caption: "Each image is fed to every neuron in the layer, and each neuron answers with a single activation. Do this for images from all three classes and a table builds up: a column per image, a row per neuron." },
  { id: "distributions", label: "Distributions", title: "One neuron, three distributions", dwell: 11,
    caption: "Zoom in on a single neuron and collect every activation it produced, colored by the class of the image. Each class leaves behind its own distribution." },
  { id: "jm", label: "JM", title: "One number per class pair", dwell: 17,
    caption: "Take two distributions at a time. Overlap is the whole question: distributions that pull apart are easy to separate, distributions that sit on top of each other are not. Jeffries–Matusita turns that overlap into one bounded number — and the three of them together become this neuron’s vector." },
  { id: "matrix", label: "Matrix", title: "One vector per neuron", dwell: 18,
    caption: "The same three numbers for every other neuron in the layer. Stack them and you have the layer separability matrix — one row per neuron, one column per class pair. This is the object CoSeP actually reasons about." },
  { id: "space", label: "Space", title: "Redundancy becomes distance", dwell: 19,
    caption: "Read each row as coordinates and every neuron becomes a point. Position now means something, and distance between two points is how differently the pair of neurons behaves." },
  { id: "mss", label: "MSS", title: "How many groups is the layer actually made of?", dwell: 26,
    caption: "Cluster the space with k-medoids for every candidate k and score each clustering with the Mean Simplified Silhouette. The score keeps creeping up as k grows, so CoSeP fits a degree-2 polynomial to the curve and takes its knee: that is how many neurons this layer keeps. No compression ratio was ever supplied." },
  { id: "select", label: "Select", title: "One survivor per group", dwell: 18,
    caption: "These k* groups are the layer’s interchangeable sets: neurons inside a group separate the classes in much the same way. From each group CoSeP keeps the member with the largest weight norm — not necessarily the medoid at its center — so the survivors cover the whole space." },
  { id: "prune", label: "Prune", title: "Remove the rest, briefly fine-tune", dwell: 12,
    caption: "Everything outside the retained set goes. A short fine-tune (2 epochs on 25% of the training data for CIFAR, 3 for ImageNet) lets the survivors adapt, and then the next layer is examined in exactly the same way." },
];

// ---------- data ----------
const CLASSES = DATA.classes;
const PAIRS = DATA.pairs;
const CH = DATA.channels;
const DEMO = DATA.demo;
const MSS = DATA.mss;
const N_CH = DATA.meta.nChannels;
const K_STAR = MSS.kStar;
const RETAINED = new Set(DATA.retained);
const thumbUrl = (c, slot) => asset(`cifar/${CLASSES[c].name}-${(slot % 4) + 1}.png`);

// ---------- small helpers ----------
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const fmt = (v, d = 2) => v.toFixed(d);
// JM value (0..2) -> teal intensity; shared by the profile, the matrix and the space transition.
const cellFill = (v) => `rgba(13,148,136,${(0.06 + (v / 2) * 0.72).toFixed(3)})`;
function rnd(n) {
  const t = Math.sin(n * 12.9898) * 43758.5453;
  return t - Math.floor(t);
}

function Txt({ x, y, size, color = C.muted, anchor = "start", weight = 400, font = SANS, opacity, rotate, children }) {
  return h("text", {
    x, y, fontSize: size, fill: color, textAnchor: anchor, fontWeight: weight, fontFamily: font, opacity,
    transform: rotate ? `rotate(${rotate} ${x} ${y})` : undefined,
  }, children);
}

// Inline math for figure labels: parts are [kind, text]; kinds: i (italic), b (bold), cal, t (sans), sub, subi, sup.
function MathText({ x, y, size, color = C.ink, anchor = "start", opacity, weight, rotate, parts }) {
  let shift = 0;
  const kids = parts.map(([kind, str], i) => {
    const p = { key: i };
    let target = 0;
    if (kind === "i") Object.assign(p, { fontFamily: KMATH, fontStyle: "italic" });
    else if (kind === "b") Object.assign(p, { fontFamily: KMAIN, fontWeight: 700 });
    else if (kind === "cal") p.fontFamily = KCAL;
    else if (kind === "t") Object.assign(p, { fontFamily: SANS, fontWeight: weight });
    else if (kind === "sub" || kind === "subi") {
      Object.assign(p, { fontFamily: kind === "subi" ? KMATH : KMAIN, fontSize: +(size * 0.72).toFixed(2) });
      if (kind === "subi") p.fontStyle = "italic";
      target = size * 0.3;
    } else if (kind === "sup") {
      Object.assign(p, { fontFamily: KMAIN, fontSize: +(size * 0.72).toFixed(2) });
      target = -size * 0.42;
    } else p.fontFamily = KMAIN;
    if (Math.abs(target - shift) > 1e-6) {
      p.dy = +(target - shift).toFixed(2);
      shift = target;
    }
    return h("tspan", p, str);
  });
  return h("text", {
    x, y, fontSize: size, fill: color, textAnchor: anchor, opacity,
    transform: rotate ? `rotate(${rotate} ${x} ${y})` : undefined,
  }, kids);
}

function Thumb({ x, y, s, cls, slot, rot = 0, opacity = 1, sw = 2 }) {
  return h("g", { transform: `translate(${x.toFixed(1)} ${y.toFixed(1)})${rot ? ` rotate(${rot.toFixed(2)} ${s / 2} ${s / 2})` : ""}`, opacity },
    h("rect", { width: s, height: s, rx: s * 0.06, fill: "#fff" }),
    h("image", { href: thumbUrl(cls, slot), width: s, height: s, preserveAspectRatio: "xMidYMid slice", clipPath: "url(#ms-clip-thumb)" }),
    h("rect", { width: s, height: s, rx: s * 0.06, fill: "none", stroke: CLS[cls].main, strokeWidth: sw }));
}

function Defs() {
  return h("defs", null,
    h("clipPath", { id: "ms-clip-thumb", clipPathUnits: "objectBoundingBox" }, h("rect", { width: 1, height: 1, rx: 0.06, ry: 0.06 })),
    h("marker", { id: "ms-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" },
      h("path", { d: "M0 0 L10 5 L0 10 z", fill: C.line })));
}

function Panel({ R, label, L, opacity = 1 }) {
  return h("g", { opacity },
    h("rect", { x: R.x, y: R.y, width: R.w, height: R.h, rx: 12, fill: C.panel, stroke: C.panelEdge }),
    label == null ? null : typeof label === "string"
      ? h(Txt, { x: R.x + 16, y: R.y + 28, size: L.fs(13), color: C.muted }, label)
      : h(MathText, { x: R.x + 16, y: R.y + 28, size: L.fs(14), color: C.ink, parts: label }));
}

function Card({ x, y, w, hgt, opacity = 1, children }) {
  return h("g", { opacity },
    h("rect", { x, y, width: w, height: hgt, rx: 12, fill: "#fff", stroke: C.panelEdge }), children);
}

// Gaussian curves on an activation axis z in [0, zmax].
function gaussAxis(x0, x1, base, peak, zmax = 2.7) {
  const zx = (z) => x0 + (z / zmax) * (x1 - x0);
  const curve = (mu, sigma) => {
    const top = gaussPdf(mu, mu, sigma);
    const pts = [];
    for (let i = 0; i <= 120; i++) {
      const z = (i / 120) * zmax;
      pts.push(`${i ? "L" : "M"}${zx(z).toFixed(1)} ${(base - (gaussPdf(z, mu, sigma) / top) * peak).toFixed(1)}`);
    }
    return pts.join(" ");
  };
  const close = `L${x1} ${base} L${x0} ${base} Z`;
  const area = (mu, sigma) => `${curve(mu, sigma)} ${close}`;
  const overlap = (g1, g2) => {
    const t1 = gaussPdf(g1.mu, g1.mu, g1.sigma), t2 = gaussPdf(g2.mu, g2.mu, g2.sigma);
    const pts = [];
    for (let i = 0; i <= 140; i++) {
      const z = (i / 140) * zmax;
      const v = Math.min(gaussPdf(z, g1.mu, g1.sigma) / t1, gaussPdf(z, g2.mu, g2.sigma) / t2);
      pts.push(`${i ? "L" : "M"}${zx(z).toFixed(1)} ${(base - v * peak).toFixed(1)}`);
    }
    return `${pts.join(" ")} ${close}`;
  };
  return { zx, curve, area, overlap, x0, x1, base, peak };
}

function ZAxis({ A, L, tickY, label }) {
  return h("g", null,
    h("line", { x1: A.x0, y1: A.base, x2: A.x1, y2: A.base, stroke: C.axis, strokeWidth: 1.2 }),
    [0, 0.5, 1, 1.5, 2, 2.5].map((t) => h("g", { key: t },
      h("line", { x1: A.zx(t), y1: A.base, x2: A.zx(t), y2: A.base + 5, stroke: C.axis }),
      h(Txt, { x: A.zx(t), y: tickY, size: L.fs(12), color: C.muted, anchor: "middle", font: MONO }, t.toFixed(1)))),
    label ? h(MathText, { x: label[0], y: label[1], size: L.fs(13), color: C.muted, anchor: "end",
      parts: [["t", "activation  "], ["i", "z"], ["sub", "ℓ,"], ["subi", "c"]] }) : null);
}
/* EQUATIONS:BEGIN (pre-rendered KaTeX HTML: wide and phone variants, plus the paper reference) */
const EQ = {"inputs":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><msub><mi mathvariant=\"script\">D</mi><mrow><mi mathvariant=\"normal\">c</mi><mi mathvariant=\"normal\">a</mi><mi mathvariant=\"normal\">l</mi></mrow></msub><mo>=</mo><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">{</mo><mo stretchy=\"false\">(</mo><munder><munder><msub><mi>x</mi><mi>n</mi></msub><mo stretchy=\"true\">⏟</mo></munder><mtext>image</mtext></munder><mo separator=\"true\">,</mo><mtext>  </mtext><munder><munder><msub><mi>y</mi><mi>n</mi></msub><mo stretchy=\"true\">⏟</mo></munder><mtext>its class</mtext></munder><mo stretchy=\"false\">)</mo><msubsup><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">}</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></msubsup></mrow><annotation encoding=\"application/x-tex\">\\mathcal{D}_{\\mathrm{cal}}=\\bigl\\{(\\underbrace{x_n}_{\\text{image}},\\;\\underbrace{y_n}_{\\text{its class}})\\bigr\\}_{n=1}^{N}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:0.8333em;vertical-align:-0.15em;\"></span><span class=\"mord\"><span class=\"mord mathcal\" style=\"margin-right:0.0278em;\">D</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.0278em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mord mathrm mtight\">cal</span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.6828em;vertical-align:-1.6016em;\"></span><span class=\"mopen\"><span class=\"delimsizing size1\">{</span></span><span class=\"mopen\">(</span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4306em;\"><span style=\"top:-1.5345em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">image</span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4306em;\"><span class=\"svg-align\" style=\"top:-2.202em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">x</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.1514em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mathnormal mtight\">n</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.798em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.6016em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4306em;\"><span style=\"top:-1.4715em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">its class</span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4306em;\"><span class=\"svg-align\" style=\"top:-2.1576em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.0359em;\">y</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.1514em;\"><span style=\"top:-2.55em;margin-left:-0.0359em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mathnormal mtight\">n</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.8424em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.5285em;\"><span></span></span></span></span></span><span class=\"mclose\">)</span><span class=\"mclose\"><span class=\"mclose\"><span class=\"delimsizing size1\">}</span></span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0812em;\"><span style=\"top:-2.3003em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">n</span><span class=\"mrel mtight\">=</span><span class=\"mord mtight\">1</span></span></span></span><span style=\"top:-3.3029em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\" style=\"margin-right:0.109em;\">N</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3997em;\"><span></span></span></span></span></span></span></span></span></span></span>","narrow":null,"ref":"Labeled calibration set (§3.1)"},"forward":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><mrow><msub><mi>h</mi><mi mathvariant=\"normal\">ℓ</mi></msub><mo stretchy=\"false\">(</mo><mi>x</mi><mo stretchy=\"false\">)</mo></mrow><mo stretchy=\"true\">⏟</mo></munder><mrow><mtext>output of layer </mtext><mi mathvariant=\"normal\">ℓ</mi></mrow></munder><mo>∈</mo><msup><mi mathvariant=\"double-struck\">R</mi><msub><mi>N</mi><mi mathvariant=\"normal\">ℓ</mi></msub></msup><mo separator=\"true\">,</mo><mspace width=\"2em\"/><munder><munder><mrow><msub><mi>N</mi><mi mathvariant=\"normal\">ℓ</mi></msub><mo>=</mo><mn>64</mn></mrow><mo stretchy=\"true\">⏟</mo></munder><mtext>neurons here</mtext></munder></mrow><annotation encoding=\"application/x-tex\">\\underbrace{h_\\ell(x)}_{\\text{output of layer }\\ell}\\in\\mathbb{R}^{N_\\ell},\\qquad\\underbrace{N_\\ell=64}_{\\text{neurons here}}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.4702em;vertical-align:-1.7202em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.4159em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">output of layer </span></span><span class=\"mord mtight\">ℓ</span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.102em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">h</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span><span class=\"mopen\">(</span><span class=\"mord mathnormal\">x</span><span class=\"mclose\">)</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.898em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.7202em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">∈</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.3754em;vertical-align:-1.4841em;\"></span><span class=\"mord\"><span class=\"mord mathbb\">R</span><span class=\"msupsub\"><span class=\"vlist-t\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.8913em;\"><span style=\"top:-3.113em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\" style=\"margin-right:0.109em;\">N</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3448em;\"><span style=\"top:-2.3488em;margin-left:-0.109em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.1512em;\"><span></span></span></span></span></span></span></span></span></span></span></span></span></span></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:2em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6833em;\"><span style=\"top:-1.5159em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">neurons here</span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6833em;\"><span class=\"svg-align\" style=\"top:-2.202em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.109em;\">N</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.109em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mord\">64</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.798em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.4841em;\"><span></span></span></span></span></span></span></span></span></span>","narrow":null,"ref":"Layer output (§3.1)"},"activations":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><mrow><msub><mi>z</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo stretchy=\"false\">(</mo><mi>x</mi><mo stretchy=\"false\">)</mo></mrow><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>activation</mtext></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mtext>of neuron </mtext><mi>c</mi></mrow></mstyle></mtd></mtr></mtable></mstyle></munder><mo>=</mo><munder><munder><mrow><msub><mi>h</mi><mi mathvariant=\"normal\">ℓ</mi></msub><mo stretchy=\"false\">(</mo><mi>x</mi><msub><mo stretchy=\"false\">)</mo><mi>c</mi></msub></mrow><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mi>c</mi><mtext>-th output</mtext></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mtext>of layer </mtext><mi mathvariant=\"normal\">ℓ</mi></mrow></mstyle></mtd></mtr></mtable></mstyle></munder></mrow><annotation encoding=\"application/x-tex\">\\underbrace{z_{\\ell,c}(x)}_{\\substack{\\text{activation}\\\\\\text{of neuron }c}}=\\underbrace{h_\\ell(x)_c}_{\\substack{c\\text{-th output}\\\\\\text{of layer }\\ell}}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.0897em;vertical-align:-2.3397em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.0881em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1111em;\"><span style=\"top:-3.1433em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">activation</span></span></span></span><span style=\"top:-2.2689em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">of neuron </span></span><span class=\"mord mathnormal mtight\">c</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6112em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.0659em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.044em;\">z</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.044em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mopen\">(</span><span class=\"mord mathnormal\">x</span><span class=\"mclose\">)</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9341em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.3397em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.0924em;vertical-align:-2.3424em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.1048em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1389em;\"><span style=\"top:-3.1444em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span><span class=\"mord text mtight\"><span class=\"mord mtight\">-th output</span></span></span></span><span style=\"top:-2.2556em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">of layer </span></span><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6389em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.102em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">h</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span><span class=\"mopen\">(</span><span class=\"mord mathnormal\">x</span><span class=\"mclose\"><span class=\"mclose\">)</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.1514em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mathnormal mtight\">c</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.898em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.3424em;\"><span></span></span></span></span></span></span></span></span></span>","narrow":null,"ref":"Eq. 2 · a convolutional channel’s feature map is first averaged over space (Eq. 1)"},"distributions":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><mrow><msub><mi>z</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo>∣</mo><mi>y</mi><mo>=</mo><mi>t</mi></mrow><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>activations</mtext></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mtext>on class </mtext><mi>t</mi></mrow></mstyle></mtd></mtr></mtable></mstyle></munder><mo>≈</mo><mi mathvariant=\"script\">N</mi><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">(</mo><munder><munder><msubsup><mi>μ</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow><mrow><mo stretchy=\"false\">(</mo><mi>t</mi><mo stretchy=\"false\">)</mo></mrow></msubsup><mo stretchy=\"true\">⏟</mo></munder><mtext>mean</mtext></munder><mo separator=\"true\">,</mo><munder><munder><mrow><mo stretchy=\"false\">(</mo><msubsup><mi>σ</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow><mrow><mo stretchy=\"false\">(</mo><mi>t</mi><mo stretchy=\"false\">)</mo></mrow></msubsup><msup><mo stretchy=\"false\">)</mo><mn>2</mn></msup></mrow><mo stretchy=\"true\">⏟</mo></munder><mtext>variance</mtext></munder><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">)</mo></mrow><annotation encoding=\"application/x-tex\">\\underbrace{z_{\\ell,c}\\mid y=t}_{\\substack{\\text{activations}\\\\\\text{on class }t}}\\approx\\mathcal{N}\\bigl(\\underbrace{\\mu^{(t)}_{\\ell,c}}_{\\text{mean}},\\underbrace{(\\sigma^{(t)}_{\\ell,c})^{2}}_{\\text{variance}}\\bigr)</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.0897em;vertical-align:-2.3397em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.0881em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1111em;\"><span style=\"top:-3.1433em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">activations</span></span></span></span><span style=\"top:-2.2689em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">on class </span></span><span class=\"mord mathnormal mtight\">t</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6112em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.0659em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.044em;\">z</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.044em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">∣</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mord mathnormal\" style=\"margin-right:0.0359em;\">y</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mord mathnormal\">t</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9341em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.3397em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">≈</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.7977em;vertical-align:-1.7529em;\"></span><span class=\"mord mathcal\" style=\"margin-right:0.1474em;\">N</span><span class=\"mopen\"><span class=\"delimsizing size1\">(</span></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-1.458em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">mean</span></span></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span class=\"svg-align\" style=\"top:-1.9594em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">μ</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-2.3987em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span><span style=\"top:-3.2198em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mopen mtight\">(</span><span class=\"mord mathnormal mtight\">t</span><span class=\"mclose mtight\">)</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4374em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0854em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.5868em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-1.2919em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">variance</span></span></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span class=\"svg-align\" style=\"top:-1.9594em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"mord\"><span class=\"mopen\">(</span><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.0359em;\">σ</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-2.3987em;margin-left:-0.0359em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span><span style=\"top:-3.2198em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mopen mtight\">(</span><span class=\"mord mathnormal mtight\">t</span><span class=\"mclose mtight\">)</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4374em;\"><span></span></span></span></span></span></span><span class=\"mclose\"><span class=\"mclose\">)</span><span class=\"msupsub\"><span class=\"vlist-t\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.8641em;\"><span style=\"top:-3.113em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">2</span></span></span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0854em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.7529em;\"><span></span></span></span></span></span><span class=\"mclose\"><span class=\"delimsizing size1\">)</span></span></span></span></span></span>","narrow":null,"ref":"Eqs. 3–4"},"jm":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><msubsup><mi>d</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow><mrow><mo stretchy=\"false\">(</mo><mi>i</mi><mo separator=\"true\">,</mo><mi>j</mi><mo stretchy=\"false\">)</mo></mrow></msubsup><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>separability</mtext></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mtext>of </mtext><mi>i</mi><mo separator=\"true\">,</mo><mspace width=\"0.1952em\"/><mi>j</mi></mrow></mstyle></mtd></mtr></mtable></mstyle></munder><mo>=</mo><mn>2</mn><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">(</mo><mn>1</mn><mo>−</mo><mi>exp</mi><mo>⁡</mo><mo stretchy=\"false\">(</mo><mo>−</mo><munder><munder><msubsup><mrow><mi mathvariant=\"normal\">B</mi><mi mathvariant=\"normal\">D</mi></mrow><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow><mrow><mo stretchy=\"false\">(</mo><mi>i</mi><mo separator=\"true\">,</mo><mi>j</mi><mo stretchy=\"false\">)</mo></mrow></msubsup><mo stretchy=\"true\">⏟</mo></munder><mpadded lspace=\"-0.5width\" width=\"0px\"><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>Bhattacharyya</mtext></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>distance</mtext></mstyle></mtd></mtr></mtable></mstyle></mpadded></munder><mo stretchy=\"false\">)</mo><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">)</mo></mrow><annotation encoding=\"application/x-tex\">\\underbrace{d^{(i,j)}_{\\ell,c}}_{\\substack{\\text{separability}\\\\\\text{of }i,\\,j}}=2\\bigl(1-\\exp(-\\underbrace{\\mathrm{BD}^{(i,j)}_{\\ell,c}}_{\\mathclap{\\substack{\\text{Bhattacharyya}\\\\\\text{distance}}}})\\bigr)</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.5746em;vertical-align:-2.5298em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-0.9622em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1389em;\"><span style=\"top:-3.1444em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">separability</span></span></span></span><span style=\"top:-2.2556em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">of </span></span><span class=\"mord mathnormal mtight\">i</span><span class=\"mpunct mtight\">,</span><span class=\"mspace mtight\" style=\"margin-right:0.1952em;\"></span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0572em;\">j</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6389em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span class=\"svg-align\" style=\"top:-1.9594em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">d</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-2.3987em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span><span style=\"top:-3.2198em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mopen mtight\">(</span><span class=\"mord mathnormal mtight\">i</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0572em;\">j</span><span class=\"mclose mtight\">)</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4374em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0854em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.5298em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:1.2em;vertical-align:-0.35em;\"></span><span class=\"mord\">2</span><span class=\"mopen\"><span class=\"delimsizing size1\">(</span></span><span class=\"mord\">1</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mbin\">−</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.5645em;vertical-align:-2.5197em;\"></span><span class=\"mop\">exp</span><span class=\"mopen\">(</span><span class=\"mord\">−</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-0.9672em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord katex-vbox mtight\"><span class=\"katex-thinbox mtight\"><span class=\"clap mtight\"><span class=\"katex-strut\" style=\"height:1.7633em;vertical-align:-0.6317em;\"></span><span class=\"katex-inner mtight\"><span><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1317em;\"><span style=\"top:-3.1372em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">Bhattacharyya</span></span></span></span><span style=\"top:-2.2483em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">distance</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6317em;\"><span></span></span></span></span></span></span></span></span></span></span><span class=\"katex-fix\"></span></span></span></span></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span class=\"svg-align\" style=\"top:-1.9594em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathrm\">BD</span></span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-2.3987em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span><span style=\"top:-3.2198em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mopen mtight\">(</span><span class=\"mord mathnormal mtight\">i</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0572em;\">j</span><span class=\"mclose mtight\">)</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4374em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0854em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.5197em;\"><span></span></span></span></span></span><span class=\"mclose\">)</span><span class=\"mclose\"><span class=\"delimsizing size1\">)</span></span></span></span></span></span>","narrow":null,"ref":"Eqs. 5–6 · bounded in [0, 2]"},"matrix":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><msub><mi mathvariant=\"bold\">s</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mtext>row </mtext><mi>c</mi><mtext> of</mtext></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>the matrix</mtext></mstyle></mtd></mtr></mtable></mstyle></munder><mo>=</mo><mo fence=\"true\" stretchy=\"true\" minsize=\"1.8em\" maxsize=\"1.8em\">[</mo><mtext> </mtext><munder><munder><msubsup><mi>d</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow><mrow><mo stretchy=\"false\">(</mo><mi>i</mi><mo separator=\"true\">,</mo><mi>j</mi><mo stretchy=\"false\">)</mo></mrow></msubsup><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>JM score</mtext></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>per pair</mtext></mstyle></mtd></mtr></mtable></mstyle></munder><mtext> </mtext><msub><mo fence=\"true\" stretchy=\"true\" minsize=\"1.8em\" maxsize=\"1.8em\">]</mo><mrow><mo stretchy=\"false\">(</mo><mi>i</mi><mo separator=\"true\">,</mo><mi>j</mi><mo stretchy=\"false\">)</mo><mo>∈</mo><msub><mi mathvariant=\"script\">P</mi><mi mathvariant=\"normal\">ℓ</mi></msub></mrow></msub></mrow><annotation encoding=\"application/x-tex\">\\underbrace{\\mathbf{s}_{\\ell,c}}_{\\substack{\\text{row }c\\text{ of}\\\\\\text{the matrix}}}=\\Bigl[\\,\\underbrace{d^{(i,j)}_{\\ell,c}}_{\\substack{\\text{JM score}\\\\\\text{per pair}}}\\,\\Bigr]_{(i,j)\\in\\mathcal{P}_\\ell}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.8028em;vertical-align:-2.3583em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4444em;\"><span style=\"top:-1.0788em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1244em;\"><span style=\"top:-3.13em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">row </span></span><span class=\"mord mathnormal mtight\">c</span><span class=\"mord text mtight\"><span class=\"mord mtight\"> of</span></span></span></span><span style=\"top:-2.2556em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">the matrix</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6244em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4444em;\"><span class=\"svg-align\" style=\"top:-2.0659em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathbf\">s</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9341em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.3583em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.6434em;vertical-align:-2.4934em;\"></span><span class=\"mopen\"><span class=\"delimsizing size2\">[</span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-0.9804em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1128em;\"><span style=\"top:-3.1295em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">JM score</span></span></span></span><span style=\"top:-2.2816em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">per pair</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6128em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span class=\"svg-align\" style=\"top:-1.9594em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3.0448em;\"><span class=\"pstrut\" style=\"height:3.0448em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">d</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0448em;\"><span style=\"top:-2.3987em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span><span style=\"top:-3.2198em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mopen mtight\">(</span><span class=\"mord mathnormal mtight\">i</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0572em;\">j</span><span class=\"mclose mtight\">)</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4374em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0854em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.4934em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mclose\"><span class=\"mclose\"><span class=\"delimsizing size2\">]</span></span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:-0.1747em;\"><span style=\"top:-2.0003em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mopen mtight\">(</span><span class=\"mord mathnormal mtight\">i</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0572em;\">j</span><span class=\"mclose mtight\">)</span><span class=\"mrel mtight\">∈</span><span class=\"mord mtight\"><span class=\"mord mathcal mtight\" style=\"margin-right:0.0822em;\">P</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3448em;\"><span style=\"top:-2.3488em;margin-left:-0.0822em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.1512em;\"><span></span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.8747em;\"><span></span></span></span></span></span></span></span></span></span></span>","narrow":null,"ref":"Eq. 7 · one column per class pair (three pairs here)"},"space":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><mrow><mi>δ</mi><mo stretchy=\"false\">(</mo><msub><mi mathvariant=\"bold\">s</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo separator=\"true\">,</mo><msub><mi mathvariant=\"bold\">s</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><msup><mi>c</mi><mo mathvariant=\"normal\" lspace=\"0em\" rspace=\"0em\">′</mo></msup></mrow></msub><mo stretchy=\"false\">)</mo></mrow><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>how differently</mtext></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mi>c</mi><mtext> and </mtext><msup><mi>c</mi><mo mathvariant=\"normal\" lspace=\"0em\" rspace=\"0em\">′</mo></msup><mtext> behave</mtext></mrow></mstyle></mtd></mtr></mtable></mstyle></munder><mo>=</mo><munder><munder><mrow><mo stretchy=\"false\">∥</mo><msub><mi mathvariant=\"bold\">s</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo>−</mo><msub><mi mathvariant=\"bold\">s</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><msup><mi>c</mi><mo mathvariant=\"normal\" lspace=\"0em\" rspace=\"0em\">′</mo></msup></mrow></msub><msub><mo stretchy=\"false\">∥</mo><mn>2</mn></msub></mrow><mo stretchy=\"true\">⏟</mo></munder><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>distance between</mtext></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mtext>profiles</mtext></mstyle></mtd></mtr></mtable></mstyle></munder></mrow><annotation encoding=\"application/x-tex\">\\underbrace{\\delta(\\mathbf{s}_{\\ell,c},\\mathbf{s}_{\\ell,c&#x27;})}_{\\substack{\\text{how differently}\\\\c\\text{ and }c&#x27;\\text{ behave}}}=\\underbrace{\\lVert\\mathbf{s}_{\\ell,c}-\\mathbf{s}_{\\ell,c&#x27;}\\rVert_2}_{\\substack{\\text{distance between}\\\\\\text{profiles}}}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.2118em;vertical-align:-2.4618em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.027em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1984em;\"><span style=\"top:-3.3317em;\"><span class=\"pstrut\" style=\"height:2.8278em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">how differently</span></span></span></span><span style=\"top:-2.3095em;\"><span class=\"pstrut\" style=\"height:2.8278em;\"></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span><span class=\"mord text mtight\"><span class=\"mord mtight\"> and </span></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span><span class=\"msupsub\"><span class=\"vlist-t\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.8278em;\"><span style=\"top:-2.931em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">′</span></span></span></span></span></span></span></span></span><span class=\"mord text mtight\"><span class=\"mord mtight\"> behave</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6984em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.0659em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.0379em;\">δ</span><span class=\"mopen\">(</span><span class=\"mord\"><span class=\"mord mathbf\">s</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mord\"><span class=\"mord mathbf\">s</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span><span class=\"msupsub\"><span class=\"vlist-t\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6828em;\"><span style=\"top:-2.786em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">′</span></span></span></span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mclose\">)</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9341em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.4618em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:3.1184em;vertical-align:-2.3684em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.0737em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1317em;\"><span style=\"top:-3.1372em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">distance between</span></span></span></span><span style=\"top:-2.2628em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">profiles</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6317em;\"><span></span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.0659em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mopen\">∥</span><span class=\"mord\"><span class=\"mord mathbf\">s</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mbin\">−</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mord\"><span class=\"mord mathbf\">s</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span><span class=\"msupsub\"><span class=\"vlist-t\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6828em;\"><span style=\"top:-2.786em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">′</span></span></span></span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mclose\"><span class=\"mclose\">∥</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3011em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">2</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9341em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.3684em;\"><span></span></span></span></span></span></span></span></span></span>","narrow":null,"ref":"Eq. 8"},"mss":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><msub><mrow><mi mathvariant=\"normal\">M</mi><mi mathvariant=\"normal\">S</mi><mi mathvariant=\"normal\">S</mi></mrow><mi mathvariant=\"normal\">ℓ</mi></msub><mo stretchy=\"false\">(</mo><mi>k</mi><mo stretchy=\"false\">)</mo><mo>=</mo><mfrac><mn>1</mn><msub><mi>N</mi><mi mathvariant=\"normal\">ℓ</mi></msub></mfrac><munder><mo>∑</mo><mi>c</mi></munder><mtext>  </mtext><munder><munder><mfrac><mrow><msub><mi>b</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo>−</mo><msub><mi>a</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub></mrow><mrow><mi>max</mi><mo>⁡</mo><mo stretchy=\"false\">{</mo><msub><mi>a</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo separator=\"true\">,</mo><mtext> </mtext><msub><mi>b</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo stretchy=\"false\">}</mo></mrow></mfrac><mo stretchy=\"true\">⏟</mo></munder><mpadded lspace=\"-0.5width\" width=\"0px\"><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mi>a</mi><mo>:</mo><mtext> to own medoid</mtext></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mi>b</mi><mo>:</mo><mtext> to nearest other medoid</mtext></mrow></mstyle></mtd></mtr></mtable></mstyle></mpadded></munder></mrow><annotation encoding=\"application/x-tex\">\\mathrm{MSS}_\\ell(k)=\\frac{1}{N_\\ell}\\sum_{c}\\;\\underbrace{\\frac{b_{\\ell,c}-a_{\\ell,c}}{\\max\\{a_{\\ell,c},\\,b_{\\ell,c}\\}}}_{\\mathclap{\\substack{a:\\ \\text{to own medoid}\\\\b:\\ \\text{to nearest other medoid}}}}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:1em;vertical-align:-0.25em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathrm\">MSS</span></span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span><span class=\"mopen\">(</span><span class=\"mord mathnormal\" style=\"margin-right:0.0315em;\">k</span><span class=\"mclose\">)</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:4.4158em;vertical-align:-3.0443em;\"></span><span class=\"mord\"><span class=\"mopen nulldelimiter\"></span><span class=\"mfrac\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3214em;\"><span style=\"top:-2.314em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.109em;\">N</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.109em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span><span style=\"top:-3.23em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"frac-line\" style=\"border-bottom-width:0.04em;\"></span></span><span style=\"top:-3.677em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\">1</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.836em;\"><span></span></span></span></span></span><span class=\"mclose nulldelimiter\"></span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mop op-limits\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.05em;\"><span style=\"top:-1.9em;margin-left:0em;\"><span class=\"pstrut\" style=\"height:3.05em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span></span></span></span><span style=\"top:-3.05em;\"><span class=\"pstrut\" style=\"height:3.05em;\"></span><span><span class=\"mop op-symbol large-op\">∑</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.25em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3714em;\"><span style=\"top:-0.7642em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord katex-vbox mtight\"><span class=\"katex-thinbox mtight\"><span class=\"clap mtight\"><span class=\"katex-strut\" style=\"height:1.7489em;vertical-align:-0.6244em;\"></span><span class=\"katex-inner mtight\"><span><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1244em;\"><span style=\"top:-3.13em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">a</span><span class=\"mrel mtight\">:</span><span class=\"mspace mtight\"><span class=\"mtight\"> </span></span><span class=\"mord text mtight\"><span class=\"mord mtight\">to own medoid</span></span></span></span><span style=\"top:-2.2556em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">b</span><span class=\"mrel mtight\">:</span><span class=\"mspace mtight\"><span class=\"mtight\"> </span></span><span class=\"mord text mtight\"><span class=\"mord mtight\">to nearest other medoid</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6244em;\"><span></span></span></span></span></span></span></span></span></span></span><span class=\"katex-fix\"></span></span></span></span></span></span></span><span style=\"top:-3.3714em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3714em;\"><span class=\"svg-align\" style=\"top:-1.7513em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3.3714em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mopen nulldelimiter\"></span><span class=\"mfrac\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3714em;\"><span style=\"top:-2.314em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mop\">max</span><span class=\"mopen\">{</span><span class=\"mord\"><span class=\"mord mathnormal\">a</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mord\"><span class=\"mord mathnormal\">b</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mclose\">}</span></span></span><span style=\"top:-3.23em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"frac-line\" style=\"border-bottom-width:0.04em;\"></span></span><span style=\"top:-3.677em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">b</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mbin\">−</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mord\"><span class=\"mord mathnormal\">a</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9721em;\"><span></span></span></span></span></span><span class=\"mclose nulldelimiter\"></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.6201em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:3.0443em;\"><span></span></span></span></span></span></span></span></span></span>","narrow":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><msub><mrow><mi mathvariant=\"normal\">M</mi><mi mathvariant=\"normal\">S</mi><mi mathvariant=\"normal\">S</mi></mrow><mi mathvariant=\"normal\">ℓ</mi></msub><mo stretchy=\"false\">(</mo><mi>k</mi><mo stretchy=\"false\">)</mo><mo>=</mo><munder><munder><mrow><mfrac><mn>1</mn><msub><mi>N</mi><mi mathvariant=\"normal\">ℓ</mi></msub></mfrac><munder><mo>∑</mo><mi>c</mi></munder><mfrac><mrow><msub><mi>b</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo>−</mo><msub><mi>a</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub></mrow><mrow><mi>max</mi><mo>⁡</mo><mo stretchy=\"false\">{</mo><msub><mi>a</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo separator=\"true\">,</mo><mtext> </mtext><msub><mi>b</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><mo stretchy=\"false\">}</mo></mrow></mfrac></mrow><mo stretchy=\"true\">⏟</mo></munder><mpadded lspace=\"-0.5width\" width=\"0px\"><mstyle scriptlevel=\"1\"><mtable rowspacing=\"0.1em\" columnalign=\"center\" columnspacing=\"1em\"><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mi>a</mi><mo>:</mo><mtext> distance to own medoid</mtext></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel=\"1\" displaystyle=\"false\"><mrow><mi>b</mi><mo>:</mo><mtext> to nearest other medoid</mtext></mrow></mstyle></mtd></mtr></mtable></mstyle></mpadded></munder></mrow><annotation encoding=\"application/x-tex\">\\mathrm{MSS}_\\ell(k)=\\underbrace{\\frac{1}{N_\\ell}\\sum_{c}\\frac{b_{\\ell,c}-a_{\\ell,c}}{\\max\\{a_{\\ell,c},\\,b_{\\ell,c}\\}}}_{\\mathclap{\\substack{a:\\ \\text{distance to own medoid}\\\\b:\\ \\text{to nearest other medoid}}}}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:1em;vertical-align:-0.25em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathrm\">MSS</span></span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span><span class=\"mopen\">(</span><span class=\"mord mathnormal\" style=\"margin-right:0.0315em;\">k</span><span class=\"mclose\">)</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:4.6937em;vertical-align:-3.3222em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3714em;\"><span style=\"top:-0.4863em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord katex-vbox mtight\"><span class=\"katex-thinbox mtight\"><span class=\"clap mtight\"><span class=\"katex-strut\" style=\"height:1.7489em;vertical-align:-0.6244em;\"></span><span class=\"katex-inner mtight\"><span><span class=\"mord mtight\"><span class=\"mord mtight\"><span class=\"mtable\"><span class=\"col-align-c\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1244em;\"><span style=\"top:-3.13em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">a</span><span class=\"mrel mtight\">:</span><span class=\"mspace mtight\"><span class=\"mtight\"> </span></span><span class=\"mord text mtight\"><span class=\"mord mtight\">distance to own medoid</span></span></span></span><span style=\"top:-2.2556em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">b</span><span class=\"mrel mtight\">:</span><span class=\"mspace mtight\"><span class=\"mtight\"> </span></span><span class=\"mord text mtight\"><span class=\"mord mtight\">to nearest other medoid</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6244em;\"><span></span></span></span></span></span></span></span></span></span></span><span class=\"katex-fix\"></span></span></span></span></span></span></span><span style=\"top:-3.3714em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3714em;\"><span class=\"svg-align\" style=\"top:-1.4734em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3.3714em;\"><span class=\"pstrut\" style=\"height:3.3714em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mopen nulldelimiter\"></span><span class=\"mfrac\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3214em;\"><span style=\"top:-2.314em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.109em;\">N</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.109em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span><span style=\"top:-3.23em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"frac-line\" style=\"border-bottom-width:0.04em;\"></span></span><span style=\"top:-3.677em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\">1</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.836em;\"><span></span></span></span></span></span><span class=\"mclose nulldelimiter\"></span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mop op-limits\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.05em;\"><span style=\"top:-1.9em;margin-left:0em;\"><span class=\"pstrut\" style=\"height:3.05em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span></span></span></span><span style=\"top:-3.05em;\"><span class=\"pstrut\" style=\"height:3.05em;\"></span><span><span class=\"mop op-symbol large-op\">∑</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.25em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mord\"><span class=\"mopen nulldelimiter\"></span><span class=\"mfrac\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.3714em;\"><span style=\"top:-2.314em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mop\">max</span><span class=\"mopen\">{</span><span class=\"mord\"><span class=\"mord mathnormal\">a</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mord\"><span class=\"mord mathnormal\">b</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mclose\">}</span></span></span><span style=\"top:-3.23em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"frac-line\" style=\"border-bottom-width:0.04em;\"></span></span><span style=\"top:-3.677em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">b</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mbin\">−</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mord\"><span class=\"mord mathnormal\">a</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9721em;\"><span></span></span></span></span></span><span class=\"mclose nulldelimiter\"></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.898em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:3.3222em;\"><span></span></span></span></span></span></span></span></span></span>","ref":"Eqs. 9–12 · Kneedle on the degree-2 fit gives k* (Eq. 13)"},"select":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><msubsup><mi>c</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>r</mi></mrow><mo lspace=\"0em\" rspace=\"0em\">⋆</mo></msubsup><mo stretchy=\"true\">⏟</mo></munder><mrow><mtext>kept from group </mtext><mi>r</mi></mrow></munder><mo>=</mo><munder><mrow><mi mathvariant=\"normal\">arg max</mi><mo>⁡</mo></mrow><mrow><mi>c</mi><mtext> </mtext><mo>∈</mo><mtext> </mtext><msub><mi mathvariant=\"script\">C</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>r</mi></mrow></msub></mrow></munder><mtext>  </mtext><munder><munder><mrow><mo stretchy=\"false\">∥</mo><msub><mi>W</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><msub><mo stretchy=\"false\">∥</mo><mn>2</mn></msub></mrow><mo stretchy=\"true\">⏟</mo></munder><mrow><mtext>weight norm </mtext><msub><mi>w</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub></mrow></munder></mrow><annotation encoding=\"application/x-tex\">\\underbrace{c^{\\star}_{\\ell,r}}_{\\text{kept from group }r}=\\operatorname*{arg\\,max}_{c\\,\\in\\,\\mathcal{C}_{\\ell,r}}\\;\\underbrace{\\lVert W_{\\ell,c}\\rVert_2}_{\\text{weight norm }w_{\\ell,c}}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.592em;vertical-align:-1.8533em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span style=\"top:-1.2828em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">kept from group </span></span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0278em;\">r</span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span class=\"svg-align\" style=\"top:-1.9689em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">c</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span style=\"top:-2.453em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0278em;\">r</span></span></span></span><span style=\"top:-3.113em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">⋆</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3831em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0311em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.8533em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.5733em;vertical-align:-1.8233em;\"></span><span class=\"mop op-limits\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4306em;\"><span style=\"top:-2.1612em;margin-left:0em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span><span class=\"mspace mtight\" style=\"margin-right:0.1952em;\"></span><span class=\"mrel mtight\">∈</span><span class=\"mspace mtight\" style=\"margin-right:0.1952em;\"></span><span class=\"mord mtight\"><span class=\"mord mathcal mtight\" style=\"margin-right:0.0583em;\">C</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3448em;\"><span style=\"top:-2.3488em;margin-left:-0.0583em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0278em;\">r</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2901em;\"><span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span><span class=\"mop\"><span class=\"mord mathrm\" style=\"margin-right:0.0139em;\">arg</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mord mathrm\">max</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1419em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.3798em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">weight norm </span></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0269em;\">w</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3448em;\"><span style=\"top:-2.3488em;margin-left:-0.0269em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2901em;\"><span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.0659em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mopen\">∥</span><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.1389em;\">W</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.1389em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mclose\"><span class=\"mclose\">∥</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3011em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">2</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9341em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.8233em;\"><span></span></span></span></span></span></span></span></span></span>","narrow":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><msubsup><mi>c</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>r</mi></mrow><mo lspace=\"0em\" rspace=\"0em\">⋆</mo></msubsup><mo stretchy=\"true\">⏟</mo></munder><mtext>kept</mtext></munder><mo>=</mo><munder><mrow><mi mathvariant=\"normal\">arg max</mi><mo>⁡</mo></mrow><mrow><mi>c</mi><mtext> </mtext><mo>∈</mo><mtext> </mtext><msub><mi mathvariant=\"script\">C</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>r</mi></mrow></msub></mrow></munder><munder><munder><mrow><mo stretchy=\"false\">∥</mo><msub><mi>W</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mi>c</mi></mrow></msub><msub><mo stretchy=\"false\">∥</mo><mn>2</mn></msub></mrow><mo stretchy=\"true\">⏟</mo></munder><mtext>weight norm</mtext></munder></mrow><annotation encoding=\"application/x-tex\">\\underbrace{c^{\\star}_{\\ell,r}}_{\\text{kept}}=\\operatorname*{arg\\,max}_{c\\,\\in\\,\\mathcal{C}_{\\ell,r}}\\underbrace{\\lVert W_{\\ell,c}\\rVert_2}_{\\text{weight norm}}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.592em;vertical-align:-1.8533em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span style=\"top:-1.2828em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">kept</span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span class=\"svg-align\" style=\"top:-1.9689em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">c</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span style=\"top:-2.453em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0278em;\">r</span></span></span></span><span style=\"top:-3.113em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">⋆</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3831em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.0311em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.8533em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.5063em;vertical-align:-1.7563em;\"></span><span class=\"mop op-limits\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4306em;\"><span style=\"top:-2.1612em;margin-left:0em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\">c</span><span class=\"mspace mtight\" style=\"margin-right:0.1952em;\"></span><span class=\"mrel mtight\">∈</span><span class=\"mspace mtight\" style=\"margin-right:0.1952em;\"></span><span class=\"mord mtight\"><span class=\"mord mathcal mtight\" style=\"margin-right:0.0583em;\">C</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3448em;\"><span style=\"top:-2.3488em;margin-left:-0.0583em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0278em;\">r</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2901em;\"><span></span></span></span></span></span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span><span class=\"mop\"><span class=\"mord mathrm\" style=\"margin-right:0.0139em;\">arg</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mord mathrm\">max</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1419em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span style=\"top:-1.3798em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">weight norm</span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.75em;\"><span class=\"svg-align\" style=\"top:-2.0659em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mopen\">∥</span><span class=\"mord\"><span class=\"mord mathnormal\" style=\"margin-right:0.1389em;\">W</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.1389em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mathnormal mtight\">c</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.2861em;\"><span></span></span></span></span></span></span><span class=\"mclose\"><span class=\"mclose\">∥</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3011em;\"><span style=\"top:-2.55em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">2</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.9341em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.7563em;\"><span></span></span></span></span></span></span></span></span></span>","ref":"Eqs. 14–15 · the strongest member, not the medoid"},"prune":{"wide":"<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><munder><munder><msub><mi mathvariant=\"script\">S</mi><mi mathvariant=\"normal\">ℓ</mi></msub><mo stretchy=\"true\">⏟</mo></munder><mtext>neurons kept</mtext></munder><mo>=</mo><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">{</mo><munder><munder><mrow><msubsup><mi>c</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><mn>1</mn></mrow><mo lspace=\"0em\" rspace=\"0em\">⋆</mo></msubsup><mo separator=\"true\">,</mo><mtext> </mtext><mo>…</mo><mo separator=\"true\">,</mo><mtext> </mtext><msubsup><mi>c</mi><mrow><mi mathvariant=\"normal\">ℓ</mi><mo separator=\"true\">,</mo><msubsup><mi>k</mi><mi mathvariant=\"normal\">ℓ</mi><mo lspace=\"0em\" rspace=\"0em\">⋆</mo></msubsup></mrow><mo lspace=\"0em\" rspace=\"0em\">⋆</mo></msubsup></mrow><mo stretchy=\"true\">⏟</mo></munder><mrow><mtext>one per group, </mtext><msubsup><mi>k</mi><mi mathvariant=\"normal\">ℓ</mi><mo lspace=\"0em\" rspace=\"0em\">⋆</mo></msubsup><mo>=</mo><mn>32</mn></mrow></munder><mo fence=\"true\" stretchy=\"true\" minsize=\"1.2em\" maxsize=\"1.2em\">}</mo></mrow><annotation encoding=\"application/x-tex\">\\underbrace{\\mathcal{S}_\\ell}_{\\text{neurons kept}}=\\bigl\\{\\underbrace{c^{\\star}_{\\ell,1},\\,\\ldots,\\,c^{\\star}_{\\ell,k^{\\star}_\\ell}}_{\\text{one per group, }k^{\\star}_\\ell=32}\\bigr\\}</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.3035em;vertical-align:-1.6202em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6833em;\"><span style=\"top:-1.5159em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">neurons kept</span></span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6833em;\"><span class=\"svg-align\" style=\"top:-2.202em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathcal\" style=\"margin-right:0.075em;\">S</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3361em;\"><span style=\"top:-2.55em;margin-left:-0.075em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\">ℓ</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.15em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.798em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.6202em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"katex-base\"><span class=\"katex-strut\" style=\"height:2.9172em;vertical-align:-2.0672em;\"></span><span class=\"mopen\"><span class=\"delimsizing size1\">{</span></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span style=\"top:-1.1758em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord text mtight\"><span class=\"mord mtight\">one per group, </span></span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0315em;\">k</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6771em;\"><span style=\"top:-2.1528em;margin-left:-0.0315em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\">ℓ</span></span></span><span style=\"top:-2.8448em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">⋆</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3472em;\"><span></span></span></span></span></span></span><span class=\"mrel mtight\">=</span><span class=\"mord mtight\">32</span></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"minner munder\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span class=\"svg-align\" style=\"top:-1.8619em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"katex-stretchy\" style=\"height:0.548em;min-width:1.6em;\"><span class=\"brace-left\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMinYMin slice\"><path d=\"M0 6l6-6h17c12.688 0 19.313.3 20 1 4 4 7.313 8.3 10 13\n 35.313 51.3 80.813 93.8 136.5 127.5 55.688 33.7 117.188 55.8 184.5 66.5.688\n 0 2 .3 4 1 18.688 2.7 76 4.3 172 5h399450v120H429l-6-1c-124.688-8-235-61.7\n-331-161C60.687 138.7 32.312 99.3 7 54L0 41V6z\"/></svg></span><span class=\"brace-center\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMidYMin slice\"><path d=\"M199572 214\nc100.7 8.3 195.3 44 280 108 55.3 42 101.7 93 139 153l9 14c2.7-4 5.7-8.7 9-14\n 53.3-86.7 123.7-153 211-199 66.7-36 137.3-56.3 212-62h199568v120H200432c-178.3\n 11.7-311.7 78.3-403 201-6 8-9.7 12-11 12-.7.7-6.7 1-18 1s-17.3-.3-18-1c-1.3 0\n-5-4-11-12-44.7-59.3-101.3-106.3-170-141s-145.3-54.3-229-60H0V214z\"/></svg></span><span class=\"brace-right\" style=\"height:0.548em;\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"400em\" height=\"0.548em\" viewBox=\"0 0 400000 548\" preserveAspectRatio=\"xMaxYMin slice\"><path d=\"M399994 0l6 6v35l-6 11c-56 104-135.3 181.3-238 232-57.3\n 28.7-117 45-179 50H-300V214h399897c43.3-7 81-15 113-26 100.7-33 179.7-91 237\n-174 2.7-5 6-9 10-13 .7-1 7.3-1 20-1h17z\"/></svg></span></span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord\"><span class=\"mord\"><span class=\"mord mathnormal\">c</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span style=\"top:-2.453em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mtight\">1</span></span></span></span><span style=\"top:-3.113em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">⋆</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3831em;\"><span></span></span></span></span></span></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"minner\">…</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mpunct\">,</span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mspace\" style=\"margin-right:0.1667em;\"></span><span class=\"mord\"><span class=\"mord mathnormal\">c</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.7387em;\"><span style=\"top:-2.453em;margin-left:0em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">ℓ</span><span class=\"mpunct mtight\">,</span><span class=\"mord mtight\"><span class=\"mord mathnormal mtight\" style=\"margin-right:0.0315em;\">k</span><span class=\"msupsub\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6771em;\"><span style=\"top:-2.1528em;margin-left:-0.0315em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\">ℓ</span></span></span><span style=\"top:-2.8448em;margin-right:0.0714em;\"><span class=\"pstrut\" style=\"height:2.5em;\"></span><span class=\"katex-sizing reset-size3 size1 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">⋆</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.3472em;\"><span></span></span></span></span></span></span></span></span></span><span style=\"top:-3.113em;margin-right:0.05em;\"><span class=\"pstrut\" style=\"height:2.7em;\"></span><span class=\"katex-sizing reset-size6 size3 mtight\"><span class=\"mord mtight\"><span class=\"mord mtight\">⋆</span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.4901em;\"><span></span></span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:1.1381em;\"><span></span></span></span></span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:2.0672em;\"><span></span></span></span></span></span><span class=\"mclose\"><span class=\"delimsizing size1\">}</span></span></span></span></span></span>","narrow":null,"ref":"Eq. 16"}};
/* EQUATIONS:END */

// ---------- the 2-D view of the separability space (Space, MSS, Select share it) ----------
function plotRect(L) {
  if (!L.narrow) return { x: 36, y: 24, w: 500, h: 372 };
  // portrait: give up some height on small screens so the explanation cards below always fit
  return { x: 12, y: 12, w: 396, h: Math.round(280 - (0.752 - Math.min(0.752, L.fscale)) * 150) };
}
const XY_X = CH.map((c) => c.xy[0]);
const XY_Y = CH.map((c) => c.xy[1]);
const XY_BOUNDS = [Math.min(...XY_X), Math.max(...XY_X), Math.min(...XY_Y), Math.max(...XY_Y)];
function plotMap(L) {
  const R = plotRect(L);
  const pad = { l: 22, r: 22, t: 46, b: 20 };
  const [x0, x1, y0, y1] = XY_BOUNDS;
  const px = (v) => R.x + pad.l + ((v - x0) / (x1 - x0)) * (R.w - pad.l - pad.r);
  const py = (v) => R.y + pad.t + (1 - (v - y0) / (y1 - y0)) * (R.h - pad.t - pad.b);
  return { R, px, py, X: (c) => px(c.xy[0]), Y: (c) => py(c.xy[1]) };
}

// Pick an anchor neuron with one close look-alike and one far, complementary partner (same rule as before).
const SPACE_PICK = (() => {
  const M = plotMap({ narrow: false });
  const prof = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
  const scr = (a, b) => Math.hypot(M.X(a) - M.X(b), M.Y(a) - M.Y(b));
  const cx = M.R.x + M.R.w * 0.62, cy = M.R.y + M.R.h * 0.42;
  let best = null;
  for (const s of CH) {
    let near = null, nearD = Infinity;
    for (const r of CH) {
      if (r.index === s.index) continue;
      const d = scr(s, r), pd = prof(s.profile, r.profile);
      if (d >= 30 && d <= 78 && pd < nearD) { nearD = pd; near = r; }
    }
    if (!near) continue;
    const far = CH.filter((e) => e.index !== s.index)
      .map((e) => ({ e, d: prof(s.profile, e.profile) }))
      .sort((a, b) => b.d - a.d).slice(0, 8)
      .reduce((a, b) => (scr(s, b.e) > scr(s, a.e) ? b : a));
    const score = (far.d - nearD) * 40 - Math.hypot(M.X(s) - cx, M.Y(s) - cy);
    if (!best || score > best.score) best = { anchor: s, near, far: far.e, score };
  }
  return best;
})();

// ---------- shapes ----------
function convexHull(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list) => {
    const out = [];
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  };
  return [...half(pts), ...half([...pts].reverse())];
}

function capsulePath(a, b, r) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (len < 1e-6) return `M${a[0] - r} ${a[1]} a${r} ${r} 0 1 0 ${2 * r} 0 a${r} ${r} 0 1 0 ${-2 * r} 0`;
  const nx = (-(b[1] - a[1]) / len) * r, ny = ((b[0] - a[0]) / len) * r;
  const f = (v) => v.toFixed(1);
  return `M${f(a[0] + nx)} ${f(a[1] + ny)} L${f(b[0] + nx)} ${f(b[1] + ny)} A${r} ${r} 0 0 0 ${f(b[0] - nx)} ${f(b[1] - ny)} ` +
    `L${f(a[0] - nx)} ${f(a[1] - ny)} A${r} ${r} 0 0 0 ${f(a[0] + nx)} ${f(a[1] + ny)} Z`;
}

// Padded, rounded outline around a group of points.
function hullPath(points, pad) {
  if (points.length === 1) return capsulePath(points[0], points[0], pad);
  const hull = points.length >= 3 ? convexHull(points) : [];
  if (hull.length < 3) {
    let a = points[0], b = points[1], best = -1;
    for (const p of points) for (const q of points) {
      const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d > best) { best = d; a = p; b = q; }
    }
    return capsulePath(a, b, pad);
  }
  const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;
  const grown = hull.map(([x, y]) => {
    const d = Math.hypot(x - cx, y - cy) || 1;
    return [x + ((x - cx) / d) * pad, y + ((y - cy) / d) * pad];
  });
  const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
  const start = mid(grown[grown.length - 1], grown[0]);
  let d = `M${start[0].toFixed(1)} ${start[1].toFixed(1)}`;
  grown.forEach((p, i) => {
    const m = mid(p, grown[(i + 1) % grown.length]);
    d += ` Q${p[0].toFixed(1)} ${p[1].toFixed(1)} ${m[0].toFixed(1)} ${m[1].toFixed(1)}`;
  });
  return `${d} Z`;
}

function starPath(cx, cy, R) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const r = i % 2 ? R * 0.45 : R;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    pts.push(`${(cx + r * Math.cos(a)).toFixed(1)} ${(cy + r * Math.sin(a)).toFixed(1)}`);
  }
  return `M${pts.join(" L")} Z`;
}

function triPath(cx, cy, s) {
  return `M${cx.toFixed(1)} ${(cy - s).toFixed(1)} L${(cx + s * 0.92).toFixed(1)} ${(cy + s * 0.62).toFixed(1)} L${(cx - s * 0.92).toFixed(1)} ${(cy + s * 0.62).toFixed(1)} Z`;
}

// Spread label/image centers apart (keeping order) so neighbours never overlap.
function spread(xs, size, gap, lo, hi) {
  const order = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const out = order.map(([x]) => Math.min(Math.max(x, lo + size / 2), hi - size / 2));
  for (let i = 1; i < out.length; i++) out[i] = Math.max(out[i], out[i - 1] + size + gap);
  const over = out[out.length - 1] - (hi - size / 2);
  if (over > 0) for (let i = 0; i < out.length; i++) out[i] -= over;
  for (let i = out.length - 2; i >= 0; i--) out[i] = Math.min(out[i], out[i + 1] - size - gap);
  const res = [];
  order.forEach(([, idx], k) => { res[idx] = out[k]; });
  return res;
}

// ---------- Step 1 · Inputs ----------
function inputsGeom(L) {
  if (!L.narrow) {
    const S = 96, gap = 12, centers = [200, 500, 800];
    const left = (c) => centers[c] - S - gap / 2;
    return {
      S, cell: (c, k) => [left(c) + (k % 2) * (S + gap), 104 + Math.floor(k / 2) * (S + gap)],
      label: (c) => [left(c), 84], noteX: 500, noteY: [368],
      note: ["Each image is 32 × 32 pixels: exactly what the network sees."],
    };
  }
  const S = 84, gap = 8;
  return {
    S, cell: (c, k) => [30 + k * (S + gap), 66 + c * 166], label: (c) => [30, 52 + c * 166],
    noteX: 210, noteY: [552, 552 + L.fs(13) * 1.35],
    note: ["Each image is 32 × 32 pixels:", "exactly what the network sees."],
  };
}

function StepInputs({ p, L }) {
  const G = inputsGeom(L);
  const fly = ease(p, 0.05, 0.62), lab = ease(p, 0.5, 0.8), note = ease(p, 0.66, 0.92);
  const fsName = L.fs(15);
  return h("g", null,
    CLASSES.map((cl, c) => {
      const [x, y] = G.label(c);
      return h("g", { key: `l${c}`, opacity: lab },
        h("circle", { cx: x + 7, cy: y - fsName * 0.34, r: 6, fill: CLS[c].main }),
        h(Txt, { x: x + 20, y, size: fsName, color: C.ink, weight: 500 }, cl.name));
    }),
    CLASSES.flatMap((cl, c) => [0, 1, 2, 3].map((k) => {
      const [tx, ty] = G.cell(c, k);
      const sx = 20 + rnd(c * 17 + k * 7) * (L.W - G.S - 40);
      const sy = 20 + rnd(c * 31 + k * 13) * (L.H - G.S - 40);
      return h(Thumb, {
        key: `${c}-${k}`, x: lerp(sx, tx, fly), y: lerp(sy, ty, fly), s: G.S, cls: c, slot: k,
        rot: lerp((rnd(c * 5 + k) - 0.5) * 24, 0, fly),
      });
    })),
    h("g", { opacity: note }, G.note.map((t, i) => h(Txt, {
      key: i, x: G.noteX, y: G.noteY[i], size: L.fs(13), color: C.muted, anchor: "middle",
    }, t))));
}

// ---------- Step 2 · Forward ----------
const LAYERS = [6, 9, 9, 7, 4];
const HL = 2; // the layer we zoom into

function forwardGeom(L) {
  if (!L.narrow) {
    return {
      card: 64, stack: (c) => [56, [64, 180, 296][c]], label: (c) => [56, [52, 168, 284][c]],
      lx: (a) => 340 + a * 120, ny: (a, j) => 210 - (LAYERS[a] - 1) * 15 + j * 30, cy: 210,
      zoom: 1.4, center: 500, r: 8, title: [60, 48], sub: [60, 78],
    };
  }
  return {
    card: 56, stack: (c) => [[34, 158, 282][c], 66], label: (c) => [[34, 158, 282][c], 54],
    lx: (a) => 50 + a * 80, ny: (a, j) => 384 - (LAYERS[a] - 1) * 13 + j * 26, cy: 384,
    zoom: 1.45, center: 210, r: 8, title: [24, 64], sub: [24, 94],
  };
}

function StepForward({ p, L }) {
  const G = forwardGeom(L), I = inputsGeom(L);
  const gather = ease(p, 0.02, 0.22), net = ease(p, 0.18, 0.36), mark = ease(p, 0.42, 0.6);
  const zoom = ease(p, 0.64, 0.9), title = ease(p, 0.88, 1);
  const mid = G.lx(HL);
  const tf = `translate(${lerp(mid, G.center, zoom).toFixed(1)} ${G.cy}) scale(${lerp(1, G.zoom, zoom).toFixed(3)}) translate(${-mid} ${-G.cy})`;
  const span = G.card + 21;
  const edges = [];
  for (let a = 0; a < LAYERS.length - 1; a++) {
    const touch = a === HL || a + 1 === HL;
    for (let j = 0; j < LAYERS[a]; j++) {
      for (let k = 0; k < LAYERS[a + 1]; k++) {
        edges.push(h("line", {
          key: `${a}-${j}-${k}`, x1: G.lx(a), y1: G.ny(a, j), x2: G.lx(a + 1), y2: G.ny(a + 1, k),
          stroke: touch ? C.accent : C.axis, strokeWidth: 0.7,
          opacity: (touch ? lerp(0.18, 0.34, mark) : 0.18) * (1 - zoom * 0.85),
        }));
      }
    }
  }
  // Nodes keep an opaque fill so faded layers never show the edges through them.
  const nodes = LAYERS.map((n, a) => {
    const on = a === HL ? mark : 0;
    return h("g", { key: a },
      Array.from({ length: n }, (_, j) => h("circle", {
        key: j, cx: G.lx(a), cy: G.ny(a, j), r: G.r, fill: on > 0 ? C.accentTint : "#fff",
        stroke: on > 0 ? C.accent : C.node, strokeWidth: lerp(1.1, 1.8, on),
        strokeOpacity: a === HL ? 1 : 1 - zoom * 0.8,
      })));
  });
  return h("g", null,
    h("g", { opacity: 1 - zoom }, CLASSES.map((cl, c) => {
      const [sx, sy] = G.stack(c), [lx, ly] = G.label(c);
      return h("g", { key: cl.name },
        [3, 2, 1, 0].map((k) => {
          const [x0, y0] = I.cell(c, k);
          return h(Thumb, {
            key: k, x: lerp(x0, sx + (3 - k) * 7, gather), y: lerp(y0, sy + (3 - k) * 7, gather),
            s: lerp(I.S, G.card, gather), cls: c, slot: k, opacity: k === 0 ? 1 : lerp(1, 0.6, gather), sw: 1.8,
          });
        }),
        h("g", { opacity: gather },
          h("circle", { cx: lx + 6, cy: ly - L.fs(13.5) * 0.34, r: 5, fill: CLS[c].main }),
          h(Txt, { x: lx + 18, y: ly, size: L.fs(13.5), color: C.ink }, cl.name)),
        h("line", {
          opacity: net, stroke: C.line, strokeWidth: 1.3, markerEnd: "url(#ms-arrow)",
          x1: L.narrow ? sx + span / 2 : sx + span + 10, y1: L.narrow ? sy + span + 8 : sy + span / 2,
          x2: L.narrow ? G.lx(0) : G.lx(0) - 22, y2: L.narrow ? G.ny(0, 0) - 22 : G.cy,
        }));
    })),
    h("g", { transform: tf, opacity: net }, edges, nodes),
    h("g", { opacity: title },
      h(MathText, { x: G.title[0], y: G.title[1], size: L.fs(17), color: C.ink, weight: 600, parts: [["t", "Linear layer "], ["r", "ℓ"]] }),
      h(MathText, { x: G.sub[0], y: G.sub[1], size: L.fs(14), color: C.muted, parts: [["i", "N"], ["sub", "ℓ"], ["r", ` = ${N_CH}`], ["t", " neurons"]] })));
}

// ---------- Step 3 · Activations ----------
function actGeom(L) {
  if (!L.narrow) {
    return {
      S: 56, colX: (a) => 222 + a * 118, imgY: 26, nx0: 500, nx: 132, rowY: (r) => 112 + r * 40, pitch: 40,
      labX: (nx) => nx - 22, labAnchor: "end", r: 10, dots: [326, 338, 350], last: 380, band: [190, 880, 36],
      from: [36, 172], foot: ["Two images per class shown; in practice, the whole calibration set."], footY: [412],
    };
  }
  return {
    S: 46, colX: (a) => 86 + a * 54, imgY: 28, nx0: 210, nx: 58, rowY: (r) => 124 + r * 54, pitch: 54,
    labX: () => 22, labAnchor: "start", r: 11, dots: [428, 440, 452], last: 486, band: [76, 412, 42],
    from: [8, 260], foot: ["Two images per class are shown;", "in practice, every image is used."],
    footY: [552, 552 + L.fs(12.5) * 1.4],
  };
}

function StepActivations({ p, L }) {
  const G = actGeom(L);
  const nN = DEMO.neurons.length, nI = DEMO.images.length;
  const setup = ease(p, 0.02, 0.12), band = ease(p, 0.88, 1), foot = ease(p, 0.84, 1);
  const win = 0.74 / nI;
  const nx = lerp(G.nx0, G.nx, setup);
  const valSize = L.fs(L.narrow ? 13 : 14);
  const labSize = L.fs(12.5);
  const right = G.colX(nI - 1) + G.S;
  return h("g", null,
    h("g", { opacity: band },
      h("rect", { x: G.band[0], y: G.rowY(0) - G.band[2] / 2, width: G.band[1] - G.band[0], height: G.band[2], rx: 8, fill: "#F1F4FE", stroke: C.accent, strokeWidth: 1.2 })),
    h("g", { opacity: setup },
      DEMO.neurons.slice(0, -1).map((id, r) => h("line", {
        key: `sep${id}`, x1: nx + G.r + 10, x2: right, y1: G.rowY(r) + G.pitch / 2, y2: G.rowY(r) + G.pitch / 2, stroke: C.grid,
      })),
      DEMO.neurons.map((id, r) => h("g", { key: id },
        h("circle", { cx: nx, cy: G.rowY(r), r: G.r, fill: C.accentTint, stroke: C.accent, strokeWidth: 1.3 }),
        h(Txt, { x: G.labX(nx), y: G.rowY(r) + labSize * 0.35, size: labSize, color: C.muted, anchor: G.labAnchor, font: MONO }, `n${r + 1}`))),
      G.dots.map((y, i) => h("circle", { key: `d${i}`, cx: nx, cy: y, r: 2.6, fill: C.accent, opacity: 0.7 })),
      h(Txt, { x: nx, y: G.last, size: labSize, color: C.muted, anchor: "middle", font: MONO }, `n${N_CH}`)),
    DEMO.images.map((im, a) => {
      const o = clamp01((p - (0.12 + a * win)) / win);
      if (o <= 0) return null;
      const fl = Math.min(1, o / 0.35), val = Math.max(0, (o - 0.3) / 0.7), flying = o < 1;
      const x = lerp(G.from[0], G.colX(a), fl), y = lerp(G.from[1], G.imgY, fl);
      return h("g", { key: a },
        flying ? DEMO.neurons.map((id, r) => h("line", {
          key: id, x1: x + G.S, y1: y + G.S / 2, x2: nx - G.r - 2, y2: G.rowY(r),
          stroke: CLS[im.classId].main, strokeWidth: 0.9, opacity: 0.28 * fl * (1 - fl * 0.5),
        })) : null,
        h(Thumb, { x, y, s: G.S, cls: im.classId, slot: im.slot, sw: flying ? 2.2 : 1.6 }),
        DEMO.neurons.map((id, r) => {
          // staggered per row, and fully opaque for every row by the end of the step
          const vi = clamp01((val - r * 0.08) / 0.5);
          if (vi <= 0) return null;
          return h(Txt, {
            key: id, x: G.colX(a) + G.S / 2, y: G.rowY(r) + valSize * 0.35, size: valSize, anchor: "middle", font: MONO,
            color: CLS[im.classId].text, opacity: 0.45 + 0.55 * vi,
          }, fmt(DEMO.values[r][a], 2));
        }));
    }),
    h("g", { opacity: foot }, G.foot.map((t, i) => h(Txt, {
      key: i, x: L.W / 2, y: G.footY[i], size: L.fs(12.5), color: C.muted, anchor: "middle",
    }, t))));
}

// ---------- Step 4 · Distributions ----------
const EXAMPLES = [
  { classId: 0, slot: 2, sample: 4 }, { classId: 0, slot: 3, sample: 21 },
  { classId: 2, slot: 2, sample: 6 }, { classId: 1, slot: 2, sample: 9 },
];
const focusSample = (c, n) => (DEMO.focusSamples[String(c)] || [])[n];

function distGeom(L) {
  if (!L.narrow) {
    return {
      A: gaussAxis(110, 890, 318, 168), tickY: 340, axisLab: [890, 370], drop: 100,
      badgeFrom: [132, 112], badge: [312, 42], title: [334, 48], sub: null, ex: { y: 92, S: 50 },
    };
  }
  return {
    A: gaussAxis(28, 392, 452, 236), tickY: 474, axisLab: [392, 502], drop: 150,
    badgeFrom: [58, 124], badge: [36, 42], title: [58, 48], sub: [22, 80], ex: { y: 104, S: 46 },
  };
}

function StepDistributions({ p, L }) {
  const G = distGeom(L), A = G.A;
  const intro = ease(p, 0.02, 0.16), fall = ease(p, 0.14, 0.46), ex = ease(p, 0.44, 0.62), gs = ease(p, 0.64, 0.86);
  const gauss = DEMO.focusGaussians;
  const exX = spread(EXAMPLES.map((e) => A.zx(focusSample(e.classId, e.sample) ?? 1)), G.ex.S, 10, A.x0, A.x1);
  const order = gauss.map((g, i) => [g.mu, i]).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
  const labSize = L.fs(13.5);
  const titleSize = L.fs(15);
  return h("g", null,
    h("circle", {
      cx: lerp(G.badgeFrom[0], G.badge[0], intro), cy: lerp(G.badgeFrom[1], G.badge[1], intro), r: lerp(10, 13, intro),
      fill: C.accentTint, stroke: C.accent, strokeWidth: 1.6,
    }),
    h("g", { opacity: intro },
      h("text", { x: G.title[0], y: G.title[1], fontSize: titleSize, fontFamily: SANS },
        h("tspan", { fill: C.ink, fontWeight: 600 }, "Neuron n1"),
        G.sub ? null : h("tspan", { fill: C.muted }, "   all its activations, colored by class")),
      G.sub ? h(Txt, { x: G.sub[0], y: G.sub[1], size: L.fs(13), color: C.muted }, "all its activations, by class") : null),
    h(ZAxis, { A, L, tickY: G.tickY, label: G.axisLab }),
    h("g", { opacity: 1 - gs * 0.45 }, CLASSES.map((cl, t) => (DEMO.focusSamples[String(t)] || []).map((v, n) => {
      const delay = ((n * 7 + t * 5) % 40) / 40;
      const o = clamp01((fall - delay * 0.4) / 0.6);
      if (o <= 0) return null;
      return h("circle", {
        key: `${t}-${n}`, cx: A.zx(v), cy: lerp(G.drop, A.base - 6 - ((n * 13) % 6) * 3, o), r: L.narrow ? 3 : 3.2,
        fill: CLS[t].main, opacity: 0.3 + 0.5 * o,
      });
    }))),
    h("g", { opacity: ex * (1 - gs) }, EXAMPLES.map((e, i) => {
      const v = focusSample(e.classId, e.sample) ?? 1;
      const dx = A.zx(v), cx = exX[i];
      return h("g", { key: i },
        h("line", { x1: dx, y1: A.base - 10, x2: cx, y2: G.ex.y + G.ex.S + 4, stroke: CLS[e.classId].main, strokeWidth: 1.2, opacity: 0.7 }),
        h("circle", { cx: dx, cy: A.base - 6, r: 5, fill: "none", stroke: CLS[e.classId].main, strokeWidth: 1.8 }),
        h(Thumb, { x: cx - G.ex.S / 2, y: G.ex.y, s: G.ex.S, cls: e.classId, slot: e.slot, sw: 1.8 }));
    })),
    h("g", { opacity: gs }, gauss.map((g, t) => {
      const rank = order.indexOf(t);
      const lift = L.narrow && rank === 1 ? labSize * 1.35 : 0;
      return h("g", { key: t },
        h("path", { d: A.area(g.mu, g.sigma), fill: CLS[t].tint, opacity: 0.75 }),
        h("path", { d: A.curve(g.mu, g.sigma), fill: "none", stroke: CLS[t].main, strokeWidth: 2.4 }),
        h("line", { x1: A.zx(g.mu), y1: A.base, x2: A.zx(g.mu), y2: A.base - A.peak - 4, stroke: CLS[t].main, strokeDasharray: "3 4", opacity: 0.6 }),
        h(Txt, { x: A.zx(g.mu), y: A.base - A.peak - 12 - lift, size: labSize, color: CLS[t].text, anchor: "middle", weight: 600 }, CLASSES[t].name));
    })));
}

// ---------- Step 5 · Jeffries–Matusita ----------
function jmGeom(L) {
  if (!L.narrow) {
    return {
      A: gaussAxis(70, 610, 280, 176), tickY: 302, axisLab: [610, 330], title: [70, 46],
      bar: { x: 70, y: 364, w: 400, h: 14, lab: [70, 352], val: [494, 377], ticksY: 400, text: "JM separability:  0 = identical,  2 = fully separated" },
      vec: { horiz: false, x: () => 712, y: (t) => 100 + t * 62, w: 190, h: 50, lab: [807, 80], cap: [807, 306] },
    };
  }
  return {
    A: gaussAxis(24, 396, 240, 160), tickY: 262, axisLab: [396, 290], title: [24, 42],
    bar: { x: 24, y: 336, w: 290, h: 14, lab: [24, 324], val: [330, 349], ticksY: 372, text: "JM separability, from 0 to 2" },
    vec: { horiz: true, x: (t) => 24 + t * 128, y: () => 432, w: 116, h: 50, lab: [210, 416], cap: [210, 514] },
  };
}

function StepJM({ p, L }) {
  const G = jmGeom(L), A = G.A, B = G.bar, V = G.vec;
  const gauss = DEMO.focusGaussians, prof = DEMO.profiles[0];
  const intro = ease(p, 0.01, 0.08);
  const win = 0.82 / PAIRS.length;
  const cur = Math.min(PAIRS.length - 1, Math.max(0, Math.floor((p - 0.08) / win)));
  const d = clamp01((p - (0.08 + cur * win)) / win);
  const shade = Math.min(1, d / 0.35), fill = clamp01((d - 0.3) / 0.35);
  const pair = PAIRS[cur], gi = gauss[pair.i], gj = gauss[pair.j], v = prof[cur];
  const valSize = L.fs(L.narrow ? 16 : 18);
  const x0s = PAIRS.map((_, t) => V.x(t)), y0s = PAIRS.map((_, t) => V.y(t));
  const left = Math.min(...x0s), right = Math.max(...x0s) + V.w, top = Math.min(...y0s), bottom = Math.max(...y0s) + V.h;
  return h("g", null,
    h("text", { x: G.title[0], y: G.title[1], fontSize: L.fs(16), fontFamily: SANS, fontWeight: 600 },
      h("tspan", { fill: CLS[pair.i].text }, CLASSES[pair.i].name),
      h("tspan", { fill: C.ink, fontWeight: 400 }, "  vs.  "),
      h("tspan", { fill: CLS[pair.j].text }, CLASSES[pair.j].name)),
    h(ZAxis, { A, L, tickY: G.tickY, label: G.axisLab }),
    gauss.map((g, t) => (t === pair.i || t === pair.j ? null : h("path", {
      key: `o${t}`, d: A.curve(g.mu, g.sigma), fill: "none", stroke: CLS[t].main, strokeWidth: 1.6, opacity: 0.22,
    }))),
    [pair.i, pair.j].map((t) => h("g", { key: t, opacity: intro },
      h("path", { d: A.area(gauss[t].mu, gauss[t].sigma), fill: CLS[t].tint, opacity: 0.7 }),
      h("path", { d: A.curve(gauss[t].mu, gauss[t].sigma), fill: "none", stroke: CLS[t].main, strokeWidth: 2.4 }))),
    h("path", { d: A.overlap(gi, gj), fill: C.ink, opacity: 0.14 * shade * intro }),
    h("g", { opacity: fill },
      h(Txt, { x: B.lab[0], y: B.lab[1], size: L.fs(12.5), color: C.muted }, B.text),
      h("rect", { x: B.x, y: B.y, width: B.w, height: B.h, rx: B.h / 2, fill: "#EEF0F3" }),
      h("rect", { x: B.x, y: B.y, width: Math.max(2, (v / 2) * B.w * fill), height: B.h, rx: B.h / 2, fill: C.keep }),
      h(Txt, { x: B.val[0], y: B.val[1], size: L.fs(20), color: C.ink, font: MONO, weight: 500 }, fmt(v, 3)),
      h(Txt, { x: B.x, y: B.ticksY, size: L.fs(12), color: C.muted, font: MONO }, "0"),
      h(Txt, { x: B.x + B.w, y: B.ticksY, size: L.fs(12), color: C.muted, font: MONO, anchor: "end" }, "2")),
    h(MathText, { x: V.lab[0], y: V.lab[1], size: L.fs(16), anchor: "middle", parts: [["b", "s"], ["sub", "ℓ, n1"]] }),
    PAIRS.map((pr, t) => {
      const done = t < cur || (t === cur && fill > 0.5);
      const x = V.x(t), y = V.y(t), midY = y + V.h / 2, val = prof[t];
      return h("g", { key: t },
        h("rect", {
          x, y, width: V.w, height: V.h, rx: 8, fill: done ? cellFill(val) : C.panel,
          stroke: t === cur ? C.ink : C.panelEdge, strokeWidth: t === cur ? 1.6 : 1,
        }),
        h("circle", { cx: x + 16, cy: midY, r: 5, fill: CLS[pr.i].main }),
        h("circle", { cx: x + 30, cy: midY, r: 5, fill: CLS[pr.j].main }),
        h(Txt, { x: x + V.w - 12, y: midY + valSize * 0.36, size: valSize, color: done ? C.ink : C.faint, anchor: "end", font: MONO }, done ? fmt(val, 3) : "—"));
    }),
    h("path", { d: `M${left - 6} ${top - 6} L${left - 12} ${top - 6} L${left - 12} ${bottom + 6} L${left - 6} ${bottom + 6}`, fill: "none", stroke: C.axis, strokeWidth: 1.4 }),
    h("path", { d: `M${right + 6} ${top - 6} L${right + 12} ${top - 6} L${right + 12} ${bottom + 6} L${right + 6} ${bottom + 6}`, fill: "none", stroke: C.axis, strokeWidth: 1.4 }),
    h(Txt, { x: V.cap[0], y: V.cap[1], size: L.fs(12.5), color: C.muted, anchor: "middle" }, "this neuron’s profile"));
}

// ---------- Step 6 · Matrix ----------
function matrixGeom(L) {
  if (!L.narrow) {
    return {
      x0: 300, cw: 128, y0: 88, ch: 4.6, rowY: (t) => 88 + t * 46, rowH: 38, boxX: (j) => 300 + j * 134, boxW: 126,
      start: 640, dotsY: 22, nameY: [44, 66], nLab: "side", pLabY: 410,
      note: { x: 728, rule: 712, lines: ["Every row is one neuron,", "described by what it", "can separate."] },
    };
  }
  return {
    x0: 92, cw: 104, y0: 104, ch: 5.2, rowY: (t) => 104 + t * 46, rowH: 38, boxX: (j) => 92 + j * 106, boxW: 98,
    start: 132, dotsY: 26, nameY: [50, 74], nLab: "vertical", pLabY: 466,
    note: { x: 210, rule: null, lines: ["Every row is one neuron,", "described by what it", "can separate."], ys: [514, 540, 566] },
  };
}

function StepMatrix({ p, L }) {
  const G = matrixGeom(L);
  const appear = ease(p, 0.02, 0.3), shrink = ease(p, 0.34, 0.56), fillp = ease(p, 0.5, 0.86), labels = ease(p, 0.84, 1);
  const textOut = 1 - Math.min(1, shrink * 2.5); // numbers leave before the rows squeeze together
  const n = CH.length, filled = Math.floor(fillp * n), nDemo = DEMO.neurons.length;
  const bottom = G.y0 + n * G.ch, midY = G.y0 + (n * G.ch) / 2;
  const valSize = L.fs(L.narrow ? 14 : 15), nameSize = L.fs(12.5);
  return h("g", null,
    h("g", { opacity: shrink }, PAIRS.map((pr, t) => {
      const X = G.x0 + t * G.cw + G.cw / 2;
      return h("g", { key: t },
        h("circle", { cx: X - 9, cy: G.dotsY, r: 5, fill: CLS[pr.i].main }),
        h("circle", { cx: X + 9, cy: G.dotsY, r: 5, fill: CLS[pr.j].main }),
        h(Txt, { x: X, y: G.nameY[0], size: nameSize, color: CLS[pr.i].text, anchor: "middle", weight: 500 }, CLASSES[pr.i].name),
        h(Txt, { x: X, y: G.nameY[1], size: nameSize, color: CLS[pr.j].text, anchor: "middle", weight: 500 }, CLASSES[pr.j].name));
    })),
    DEMO.neurons.map((id, t) => {
      const o = t === 0 ? 1 : clamp01((appear - t * 0.14) / 0.4);
      if (o <= 0) return null;
      const slide = t === 0 ? G.start - G.boxX(0) : 0;
      return h("g", { key: id, opacity: o },
        DEMO.profiles[t].map((v, j) => {
          const bx = lerp(G.boxX(j) + slide * (1 - appear), G.x0 + j * G.cw, shrink);
          const by = lerp(G.rowY(t), G.y0 + t * G.ch, shrink);
          const bw = lerp(G.boxW, G.cw - 1.5, shrink), bh = lerp(G.rowH, G.ch - 1.2, shrink);
          return h("g", { key: j },
            h("rect", { x: bx, y: by, width: bw, height: bh, rx: lerp(6, 0, shrink), fill: cellFill(v), stroke: C.panelEdge, strokeWidth: 0.8 * (1 - shrink) }),
            h(Txt, { x: bx + bw / 2, y: by + bh / 2 + valSize * 0.35, size: valSize, color: C.ink, anchor: "middle", font: MONO, opacity: textOut }, fmt(v, 3)));
        }),
        h(Txt, { x: G.boxX(0) + slide * (1 - appear) - 14, y: G.rowY(t) + G.rowH / 2 + nameSize * 0.35, size: nameSize, color: C.muted, anchor: "end", font: MONO, opacity: textOut }, `n${t + 1}`));
    }),
    CH.slice(nDemo, filled).map((c, k) => {
      const t = k + nDemo;
      return h("g", { key: c.index },
        c.profile.map((v, j) => h("rect", { key: j, x: G.x0 + j * G.cw, y: G.y0 + t * G.ch, width: G.cw - 1.5, height: G.ch - 1.2, fill: cellFill(v) })),
        t === filled - 1 && labels < 1 ? h("rect", { x: G.x0 - 3, y: G.y0 + t * G.ch - 1, width: 3 * G.cw + 4, height: G.ch + 1, fill: "none", stroke: C.ink, strokeWidth: 1, opacity: 0.6 }) : null);
    }),
    h("g", { opacity: labels },
      h("rect", { x: G.x0 - 4, y: G.y0 - 4, width: 3 * G.cw + 6, height: n * G.ch + 6, rx: 4, fill: "none", stroke: C.line }),
      G.nLab === "side"
        ? h("g", null,
          h(MathText, { x: G.x0 - 18, y: midY, size: L.fs(14), color: C.ink, anchor: "end", parts: [["i", "N"], ["sub", "ℓ"], ["r", ` = ${N_CH}`]] }),
          h(Txt, { x: G.x0 - 18, y: midY + L.fs(14) * 1.4, size: L.fs(12.5), color: C.muted, anchor: "end" }, "neurons"))
        : h(MathText, { x: G.x0 - 34, y: midY, size: L.fs(14), color: C.ink, anchor: "middle", rotate: -90, parts: [["i", "N"], ["sub", "ℓ"], ["r", ` = ${N_CH}`], ["t", " neurons"]] }),
      h(MathText, { x: G.x0 + 1.5 * G.cw, y: G.pLabY, size: L.fs(14), color: C.ink, anchor: "middle", parts: [["r", "|"], ["cal", "P"], ["sub", "ℓ"], ["r", "| = 3"], ["t", " class pairs"]] }),
      G.note.rule ? h("line", { x1: G.note.rule, y1: G.y0 + 40, x2: G.note.rule, y2: bottom - 40, stroke: C.grid }) : null,
      G.note.lines.map((t, i) => h(Txt, {
        key: i, x: G.note.x, y: G.note.ys ? G.note.ys[i] : midY + (i - 1) * L.fs(14) * 1.6,
        size: L.fs(14), color: C.text, anchor: G.note.ys ? "middle" : "start",
      }, t))));
}

// ---------- Step 7 · Space ----------
const SPACE_CARDS = [
  { title: "These two sit together", color: C.cut, lines: ["They separate the classes in", "much the same way, so keeping", "both adds almost nothing."] },
  { title: "These two sit apart", color: CLS[1].main, lines: ["They separate different", "classes, so they complement", "each other: keep both."] },
];
function spaceGeom(L) {
  if (!L.narrow) return { mTo: { x0: 640, cw: 64, y0: 88, ch: 4.6 }, cards: [[576, 76, 388, 124], [576, 226, 388, 124]], body: 13.5, lead: 1.6 };
  const R = plotRect(L);
  const bs = L.fs(12.5), lh = bs * 1.5;
  const hgt = Math.round(30 + lh * 1.25 + 2 * lh + bs * 0.35 + 12);
  const y0 = R.y + R.h + 10;
  return { mTo: { x0: 150, cw: 40, y0: 330, ch: 3.8 }, cards: [[12, y0, 396, hgt], [12, y0 + hgt + 12, 396, hgt]], body: 12.5, lead: 1.5 };
}

function SpaceCard({ L, G, box, card, opacity }) {
  const [x, y, w, hh] = box;
  const ts = L.fs(15), bs = L.fs(G.body), lh = bs * G.lead;
  return h(Card, { x, y, w, hgt: hh, opacity },
    h("circle", { cx: x + 24, cy: y + 30 - ts * 0.34, r: 6, fill: card.color }),
    h(Txt, { x: x + 38, y: y + 30, size: ts, color: C.ink, weight: 600 }, card.title),
    card.lines.map((t, i) => h(Txt, { key: i, x: x + 24, y: y + 30 + lh * 1.25 + i * lh, size: bs, color: C.text }, t)));
}

function StepSpace({ p, L }) {
  const G = spaceGeom(L), M0 = matrixGeom(L), M = plotMap(L);
  const frame = ease(p, 0.02, 0.16), move = ease(p, 0.18, 0.42), focus = ease(p, 0.46, 0.56), near = ease(p, 0.58, 0.72), far = ease(p, 0.76, 0.9);
  const mx = lerp(M0.x0, G.mTo.x0, frame), mcw = lerp(M0.cw, G.mTo.cw, frame), my = lerp(M0.y0, G.mTo.y0, frame), mch = lerp(M0.ch, G.mTo.ch, frame);
  const n = CH.length;
  const { anchor: A, near: N, far: F } = SPACE_PICK;
  const ax = M.X(A), ay = M.Y(A), nxp = M.X(N), nyp = M.Y(N), fx = M.X(F), fy = M.Y(F);
  const [c1x, c1y, , c1h] = G.cards[0];
  return h("g", null,
    h(Panel, { R: M.R, L, label: "each point is one neuron", opacity: frame }),
    CH.map((c, t) => {
      const o = clamp01((move - (t / n) * 0.5) / 0.5);
      if (o >= 1) return null;
      return h("g", { key: `row${c.index}`, opacity: 1 - o },
        c.profile.map((v, j) => h("rect", { key: j, x: mx + j * mcw, y: my + t * mch, width: mcw - 1.2, height: Math.max(0.6, mch - 1), fill: cellFill(v) })));
    }),
    CH.map((c, t) => {
      const o = clamp01((move - (t / n) * 0.5) / 0.5);
      if (o <= 0) return null;
      const lit = c.index === A.index || c.index === N.index || c.index === F.index;
      return h("circle", {
        key: c.index, cx: lerp(mx + 1.5 * mcw, M.X(c), o), cy: lerp(my + t * mch + mch / 2, M.Y(c), o), r: lerp(2.2, 5, o),
        fill: C.accent, opacity: (0.35 + 0.5 * o) * (lit ? 1 : 1 - 0.55 * focus),
      });
    }),
    h("g", { opacity: focus },
      h("circle", { cx: ax, cy: ay, r: lerp(18, 10, focus), fill: "none", stroke: C.ink, strokeWidth: 2 }),
      h("circle", { cx: ax, cy: ay, r: 5, fill: C.ink })),
    h("g", { opacity: near },
      h("line", { x1: ax, y1: ay, x2: nxp, y2: nyp, stroke: C.cut, strokeWidth: 2 }),
      h("circle", { cx: nxp, cy: nyp, r: 10, fill: "none", stroke: C.cut, strokeWidth: 2 }),
      h("circle", { cx: nxp, cy: nyp, r: 4.8, fill: C.cut }),
      h("line", L.narrow
        ? { x1: (ax + nxp) / 2, y1: Math.max(ay, nyp) + 14, x2: (ax + nxp) / 2, y2: c1y, stroke: C.cut, strokeWidth: 1, opacity: 0.45 }
        : { x1: Math.max(ax, nxp) + 14, y1: (ay + nyp) / 2, x2: c1x, y2: c1y + c1h / 2, stroke: C.cut, strokeWidth: 1, opacity: 0.45 })),
    h(SpaceCard, { L, G, box: G.cards[0], card: SPACE_CARDS[0], opacity: near }),
    h("g", { opacity: far },
      h("line", { x1: ax, y1: ay, x2: fx, y2: fy, stroke: CLS[1].main, strokeWidth: 1.8, strokeDasharray: "6 5" }),
      h("circle", { cx: fx, cy: fy, r: 10, fill: "none", stroke: CLS[1].main, strokeWidth: 2 }),
      h("circle", { cx: fx, cy: fy, r: 4.8, fill: CLS[1].main })),
    h(SpaceCard, { L, G, box: G.cards[1], card: SPACE_CARDS[1], opacity: far }));
}

// ---------- Step 8 · MSS and the knee ----------
const FRAMES = Object.keys(DATA.clusterSteps).map(Number).sort((a, b) => a - b);
const MSS_LO = Math.min(...MSS.values, ...MSS.smoothed) - 0.02;
const MSS_HI = Math.max(...MSS.values, ...MSS.smoothed) + 0.02;
const K0 = MSS.k[0], K1 = MSS.k[MSS.k.length - 1];
const chordAt = (k) => MSS.chord.y0 + ((k - MSS.chord.k0) / (MSS.chord.k1 - MSS.chord.k0)) * (MSS.chord.y1 - MSS.chord.y0);

function mssGeom(L) {
  const f = L.fs(12.5), g = f * 1.75; // legend font and row pitch, in viewBox units
  if (!L.narrow) {
    const big = { x: 112, y: 40, w: 788, h: 258 };
    return { small: { x: 604, y: 70, w: 336, h: 250 }, big, summary: [112, 362, 788, 46],
      legend: { x: 628, y: big.y + big.h - 12 - 3 * g, gap: g } };
  }
  // Portrait: the plot gives up height on small screens so legend and summary always fit below it.
  const bh = Math.max(200, Math.min(270, 270 - (0.752 - L.fscale) * 300));
  const big = { x: 64, y: 52, w: 336, h: bh };
  const ly = big.y + bh + 44 + g;
  const f15 = L.fs(15), f135 = L.fs(13.5);
  const sy = ly + 3 * g + 0.35 * f + 18;
  const l1 = 14 + f15 * 0.9, l2 = l1 + f135 * 1.5;
  return { small: { x: 60, y: 330, w: 340, h: 190 }, big, summary: [12, sy, 396, l2 + 14 + f135 * 0.35], lines: [l1, l2],
    legend: { x: 64, y: ly, gap: g } };
}

// Legend rows: dots = MSS for each k, solid = degree-2 fit, dashed = chord, orange = largest gap (the knee).
function MssLegend({ G, L, fitOn, gapOn }) {
  const s = L.fs(12.5), x = G.legend.x, y = G.legend.y, g = G.legend.gap;
  const row = (i, mark, text, color = C.text, op = 1) => h("g", { key: i, opacity: op },
    mark(y + i * g - s * 0.35), h(Txt, { x: x + 34, y: y + i * g, size: s, color }, text));
  return h("g", null,
    row(0, (cy) => h("circle", { cx: x + 12, cy, r: 3.2, fill: C.accent, opacity: 0.7 }), "MSS for each candidate k"),
    row(1, (cy) => h("line", { x1: x, y1: cy, x2: x + 24, y2: cy, stroke: C.ink, strokeWidth: 2.2 }), "degree-2 polynomial fit (p = 2)", C.text, fitOn),
    row(2, (cy) => h("line", { x1: x, y1: cy, x2: x + 24, y2: cy, stroke: C.axis, strokeWidth: 1.4, strokeDasharray: "5 4" }), "chord between its ends", C.text, gapOn),
    row(3, (cy) => h("line", { x1: x + 12, y1: cy - 8, x2: x + 12, y2: cy + 8, stroke: C.knee, strokeWidth: 2.6 }), "largest gap: the knee", C.kneeText, gapOn));
}

function StepMSS({ p, L }) {
  const G = mssGeom(L), M = plotMap(L);
  // The scatter is gone before the chart grows into its place, so the two never overlap.
  const frames = ease(p, 0.02, 0.32), fade = ease(p, 0.34, 0.4), grow = ease(p, 0.38, 0.5), draw = ease(p, 0.5, 0.62);
  const gap = ease(p, 0.62, 0.74), knee = ease(p, 0.74, 0.84), sum = ease(p, 0.86, 1);
  const kFrame = FRAMES[Math.min(FRAMES.length - 1, Math.floor(frames * FRAMES.length))];
  const fr = DATA.clusterSteps[String(kFrame)];
  const shown = Math.floor(ease(p, 0.06, 0.34) * MSS.k.length);
  const R = { x: lerp(G.small.x, G.big.x, grow), y: lerp(G.small.y, G.big.y, grow), w: lerp(G.small.w, G.big.w, grow), h: lerp(G.small.h, G.big.h, grow) };
  const X = (k) => R.x + ((k - K0) / (K1 - K0)) * R.w;
  const Y = (v) => R.y + R.h - ((v - MSS_LO) / (MSS_HI - MSS_LO)) * R.h;
  const bottom = R.y + R.h;
  const iStar = MSS.k.indexOf(K_STAR), xs = X(K_STAR), yFit = Y(MSS.smoothed[iStar]), yChord = Y(chordAt(K_STAR));
  const nFit = Math.max(2, Math.floor(draw * MSS.k.length));
  const fitD = MSS.k.slice(0, nFit).map((k, i) => `${i ? "L" : "M"}${X(k).toFixed(1)} ${Y(MSS.smoothed[i]).toFixed(1)}`).join(" ");
  const tick = L.fs(12), small = L.fs(12.5);
  const [sx, sy, sw, sh] = G.summary;
  const pct = ((K_STAR / N_CH) * 100).toFixed(1);
  return h("g", null,
    h("g", { opacity: 1 - fade },
      h(Panel, { R: M.R, L, label: [["i", "k"], ["r", ` = ${kFrame}`]] }),
      CH.map((c) => h("circle", { key: c.index, cx: M.X(c), cy: M.Y(c), r: 4.6, fill: groupColor(fr.assign[c.index]), opacity: 0.8 })),
      fr.medoids.map((m, ci) => h("path", { key: `m${ci}`, d: triPath(M.X(CH[m]), M.Y(CH[m]), 7), fill: groupColor(ci), stroke: "#fff", strokeWidth: 1.4 }))),
    h("line", { x1: R.x, y1: bottom, x2: R.x + R.w, y2: bottom, stroke: C.axis, strokeWidth: 1.2 }),
    h("line", { x1: R.x, y1: R.y, x2: R.x, y2: bottom, stroke: C.axis, strokeWidth: 1.2 }),
    h(MathText, { x: R.x, y: R.y - 12, size: L.fs(13), color: C.muted, parts: [["r", "MSS("], ["i", "k"], ["r", ")"]] }),
    MSS.k.slice(0, shown).map((k, i) => h("circle", { key: k, cx: X(k), cy: Y(MSS.values[i]), r: lerp(2.6, 3.2, grow), fill: C.accent, opacity: 0.65 })),
    h(Txt, { x: X(K0), y: bottom + 20, size: tick, color: C.muted, font: MONO, anchor: "middle" }, String(K0)),
    h(Txt, { x: X(K1), y: bottom + 20, size: tick, color: C.muted, font: MONO, anchor: "middle" }, String(K1)),
    h(MathText, { x: R.x + R.w / 2, y: bottom + 44, size: small, color: C.muted, anchor: "middle", opacity: grow, parts: [["t", "candidate number of groups "], ["i", "k"]] }),
    h("path", { d: fitD, fill: "none", stroke: C.ink, strokeWidth: 2.2, opacity: draw > 0 ? 0.9 : 0 }),
    h("g", { opacity: gap },
      h("line", { x1: X(MSS.chord.k0), y1: Y(MSS.chord.y0), x2: X(MSS.chord.k1), y2: Y(MSS.chord.y1), stroke: C.axis, strokeWidth: 1.4, strokeDasharray: "5 5" }),
      h("line", { x1: xs, y1: yFit, x2: xs, y2: yChord, stroke: C.knee, strokeWidth: 2.6 })),
    h("g", { opacity: grow }, h(MssLegend, { G, L, fitOn: draw, gapOn: gap })),
    h("g", { opacity: knee },
      h("line", { x1: xs, y1: R.y, x2: xs, y2: bottom, stroke: C.knee, strokeDasharray: "4 4", opacity: 0.55 }),
      h("circle", { cx: xs, cy: yFit, r: lerp(14, 7, knee), fill: C.knee }),
      h("circle", { cx: xs, cy: yFit, r: 13, fill: "none", stroke: C.knee, opacity: 0.35 }),
      h(MathText, { x: xs, y: bottom + 20, size: L.fs(14), color: C.kneeText, anchor: "middle", parts: [["i", "k"], ["sup", "⋆"], ["r", ` = ${K_STAR}`]] })),
    h(Card, { x: sx, y: sy, w: sw, hgt: sh, opacity: sum },
      L.narrow
        ? h("g", null,
          h(Txt, { x: sx + sw / 2, y: sy + G.lines[0], size: L.fs(15), color: C.ink, anchor: "middle", weight: 600 }, `Keep ${K_STAR} of ${N_CH} neurons`),
          h(Txt, { x: sx + sw / 2, y: sy + G.lines[1], size: L.fs(13.5), color: C.muted, anchor: "middle" }, `${pct}% of this layer`))
        : h(Txt, { x: sx + sw / 2, y: sy + sh / 2 + L.fs(15) * 0.36, size: L.fs(15), color: C.ink, anchor: "middle", weight: 600 }, `Keep ${K_STAR} of ${N_CH} neurons — ${pct}% of this layer`)));
}

// ---------- Step 9 · Select ----------
const GROUPS = Array.from({ length: K_STAR }, (_, ci) => {
  const members = CH.filter((c) => c.cluster === ci);
  return { ci, members, rep: members.find((m) => m.isRepresentative), med: members.find((m) => m.isMedoid) };
}).filter((g) => g.members.length > 0);
const NOT_MEDOID = GROUPS.filter((g) => g.rep !== g.med).length;
const W_MAX = Math.max(...CH.map((c) => c.weight));
// The worked example: a well-spread, uncrowded group whose kept member is not its medoid (computed on the wide view).
const FOCUS_GROUP = (() => {
  const M = plotMap({ narrow: false });
  const pt = (c) => [M.X(c), M.Y(c)];
  const scored = GROUPS.filter((g) => g.members.length >= 2 && g.rep !== g.med).map((g) => {
    const P = g.members.map(pt);
    const span = Math.max(...P.flatMap((a) => P.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1]))));
    const cx = P.reduce((s, q) => s + q[0], 0) / P.length, cy = P.reduce((s, q) => s + q[1], 0) / P.length;
    const crowd = CH.filter((c) => c.cluster !== g.ci && Math.hypot(M.X(c) - cx, M.Y(c) - cy) < 30).length;
    return { g, span, crowd };
  });
  scored.sort((a, b) => b.g.members.length - a.g.members.length || a.crowd - b.crowd || b.span - a.span);
  return scored.length ? scored[0].g : null;
})();

function selectGeom(L) {
  if (!L.narrow) return { card: [576, 24, 388, 372], x: 600, head: 58, sub: 84, row0: 124, pitch: 62, barW: 300, foot: [318, 342] };
  // portrait: stack everything by font metrics so nothing touches at any phone width
  const R = plotRect(L);
  const f15 = L.fs(15), f13 = L.fs(13), f135 = L.fs(13.5);
  const top = R.y + R.h + 10;
  const head = top + 14 + f15 * 0.95, sub = head + f13 * 1.55, row0 = sub + f13 * 1.9;
  const pitch = Math.max(46, f13 * 1.2 + 26);
  const last = row0 + 2 * pitch;
  const foot0 = last + 22 + f135 * 1.7, foot1 = foot0 + f135 * 1.45;
  return { card: [12, top, 396, foot1 + 14 - top], x: 32, head, sub, row0, pitch, barW: 300, foot: [foot0, foot1] };
}

function StepSelect({ p, L }) {
  const G = selectGeom(L), M = plotMap(L);
  const pts = ease(p, 0.02, 0.14), hulls = ease(p, 0.14, 0.36), meds = ease(p, 0.36, 0.5), stars = ease(p, 0.48, 0.64);
  const panel = ease(p, 0.62, 0.82), dim = ease(p, 0.84, 1);
  const F = FOCUS_GROUP;
  const rows = F ? [...F.members].sort((a, b) => b.weight - a.weight) : [];
  const [cx, cy, cw, chh] = G.card;
  const fp = F ? F.members.map((m) => [M.X(m), M.Y(m)]) : [];
  const fx = fp.length ? Math.max(...fp.map((q) => q[0])) + 16 : 0, fyc = fp.length ? fp.reduce((s, q) => s + q[1], 0) / fp.length : 0;
  const fxc = fp.length ? fp.reduce((s, q) => s + q[0], 0) / fp.length : 0, fyb = fp.length ? Math.max(...fp.map((q) => q[1])) + 16 : 0;
  const name = L.fs(13), tagS = L.fs(13);
  return h("g", null,
    h(Panel, { R: M.R, L, label: [["i", "k"], ["sup", "⋆"], ["r", ` = ${K_STAR}`], ["t", " groups"]] }),
    h("g", { opacity: hulls }, GROUPS.map((g) => h("path", {
      key: g.ci, d: hullPath(g.members.map((m) => [M.X(m), M.Y(m)]), 14), fill: groupColor(g.ci), fillOpacity: 0.12,
      stroke: F && g.ci === F.ci && panel > 0 ? C.ink : groupColor(g.ci), strokeOpacity: F && g.ci === F.ci && panel > 0 ? 0.35 + 0.65 * panel : 0.45,
      strokeWidth: F && g.ci === F.ci ? lerp(1, 1.8, panel) : 1,
    }))),
    CH.map((c) => h("circle", { key: c.index, cx: M.X(c), cy: M.Y(c), r: 4, fill: groupColor(c.cluster), opacity: pts * (c.isRepresentative ? 1 : 1 - 0.7 * dim) })),
    h("g", { opacity: meds * (1 - 0.5 * dim) }, GROUPS.filter((g) => g.med && g.med !== g.rep).map((g) => h("path", {
      key: g.ci, d: triPath(M.X(g.med), M.Y(g.med), 6.5), fill: "#fff", stroke: groupColor(g.ci), strokeWidth: 1.8,
    }))),
    h("g", { opacity: stars }, GROUPS.map((g) => h("path", {
      key: g.ci, d: starPath(M.X(g.rep), M.Y(g.rep), lerp(6, 9.5, stars)), fill: groupColor(g.ci), stroke: C.ink, strokeWidth: 0.9, strokeLinejoin: "round",
    }))),
    F ? h("line", L.narrow
      ? { x1: fxc, y1: fyb, x2: fxc, y2: cy, stroke: C.ink, strokeWidth: 1, opacity: 0.35 * panel }
      : { x1: fx, y1: fyc, x2: cx, y2: cy + 110, stroke: C.ink, strokeWidth: 1, opacity: 0.35 * panel }) : null,
    F ? h(Card, { x: cx, y: cy, w: cw, hgt: chh, opacity: panel },
      h(Txt, { x: G.x, y: G.head, size: L.fs(15), color: C.ink, weight: 600 }, "Inside one group"),
      h(MathText, { x: G.x, y: G.sub, size: L.fs(13), color: C.muted, parts: [["t", "weight norm "], ["r", "‖"], ["i", "W"], ["r", "‖"], ["sub", "2"], ["t", " of each member"]] }),
      rows.map((m, i) => {
        const y = G.row0 + i * G.pitch, isRep = m === F.rep, isMed = m === F.med;
        const tagX = G.x + name * 3.4;
        return h("g", { key: m.index },
          h(Txt, { x: G.x, y, size: name, color: C.ink, font: MONO, weight: 500 }, `n${m.index + 1}`),
          isRep ? h("path", { d: starPath(tagX + 7, y - tagS * 0.34, 7), fill: groupColor(F.ci), stroke: C.ink, strokeWidth: 0.9 }) : null,
          isMed ? h("path", { d: triPath(tagX + 7, y - tagS * 0.3, 6), fill: "#fff", stroke: groupColor(F.ci), strokeWidth: 1.8 }) : null,
          isRep || isMed ? h(Txt, { x: tagX + 20, y, size: tagS, color: isRep ? C.ink : C.muted, weight: isRep ? 600 : 400 },
            isRep ? (L.narrow ? "kept: largest norm" : "kept: largest weight norm") : (L.narrow ? "medoid: the center" : "medoid: the group’s center")) : null,
          h("rect", { x: G.x, y: y + 10, width: G.barW, height: 12, rx: 6, fill: "#EEF0F3" }),
          h("rect", { x: G.x, y: y + 10, width: (m.weight / W_MAX) * G.barW, height: 12, rx: 6, fill: groupColor(F.ci), opacity: isRep ? 1 : 0.55 }),
          h(Txt, { x: G.x + G.barW + 12, y: y + 21, size: L.fs(12), color: C.muted, font: MONO }, fmt(m.weight, 2)));
      }),
      h(Txt, { x: G.x, y: G.foot[0], size: L.fs(13.5), color: C.text }, "The kept neuron is not the medoid"),
      h(Txt, { x: G.x, y: G.foot[1], size: L.fs(13.5), color: C.text }, `in ${NOT_MEDOID} of the ${K_STAR} groups.`)) : null);
}

// ---------- Step 10 · Prune ----------
function pruneGeom(L) {
  if (!L.narrow) {
    return {
      title: [500, 42], cols: 16, cx: (j) => 155 + j * 46, cy: (i) => 94 + i * 46, r: 15,
      legend: { y: 292, kept: 336, pruned: 560 }, card: [150, 318, 700, 86], text: [174, 344, 370],
      lines: ["2–3 epochs on 25% of the training data, then the next", "layer goes through exactly the same steps."],
    };
  }
  return {
    title: [210, 42], cols: 8, cx: (j) => 56 + j * 44, cy: (i) => 86 + i * 44, r: 15,
    legend: { y: 442, kept: 64, pruned: 226 }, card: [12, 466, 396, 116], text: [32, 492, 518],
    lines: ["2–3 epochs on 25% of the", "training data, then the next", "layer, in exactly the same way."],
  };
}

function StepPrune({ p, L }) {
  const G = pruneGeom(L);
  const grid = ease(p, 0.02, 0.18), keep = ease(p, 0.22, 0.44), cut = ease(p, 0.48, 0.74), card = ease(p, 0.78, 1);
  const kept = RETAINED.size, pruned = N_CH - kept;
  const ls = L.fs(13.5), ly = G.legend.y;
  const [bx, by, bw, bh] = G.card;
  const lh = L.fs(14) * 1.5;
  return h("g", null,
    h(MathText, { x: G.title[0], y: G.title[1], size: L.fs(16), color: C.ink, anchor: "middle", weight: 600, parts: [["t", "Layer "], ["r", "ℓ"], ["t", `:  ${N_CH} → ${kept} neurons`]] }),
    CH.map((c, t) => {
      const isKept = RETAINED.has(c.index);
      const x = G.cx(t % G.cols), y = G.cy(Math.floor(t / G.cols));
      const d = G.r * 0.62;
      return h("g", { key: c.index, opacity: grid * (isKept ? 1 : 1 - 0.6 * cut) },
        h("circle", {
          cx: x, cy: y, r: G.r, fill: isKept && keep > 0 ? C.keepTint : isKept ? "#fff" : "#F3F4F6",
          stroke: isKept && keep > 0 ? C.keep : C.line, strokeWidth: isKept ? lerp(1, 1.8, keep) : 1,
        }),
        !isKept && cut > 0.05 ? h("g", { opacity: cut },
          h("line", { x1: x - d, y1: y - d, x2: x + d, y2: y + d, stroke: C.cut, strokeWidth: 1.5, opacity: 0.8 }),
          h("line", { x1: x + d, y1: y - d, x2: x - d, y2: y + d, stroke: C.cut, strokeWidth: 1.5, opacity: 0.8 })) : null);
    }),
    h("g", { opacity: keep },
      h("circle", { cx: G.legend.kept, cy: ly - ls * 0.34, r: 7, fill: C.keepTint, stroke: C.keep, strokeWidth: 1.6 }),
      h(Txt, { x: G.legend.kept + 14, y: ly, size: ls, color: C.text }, `kept — ${kept}`)),
    h("g", { opacity: cut },
      h("circle", { cx: G.legend.pruned, cy: ly - ls * 0.34, r: 7, fill: "#F3F4F6", stroke: C.line }),
      h("path", { d: `M${G.legend.pruned - 4} ${ly - ls * 0.34 - 4} l8 8 M${G.legend.pruned + 4} ${ly - ls * 0.34 - 4} l-8 8`, stroke: C.cut, strokeWidth: 1.4 }),
      h(Txt, { x: G.legend.pruned + 14, y: ly, size: ls, color: C.text }, `pruned — ${pruned}`)),
    h(Card, { x: bx, y: by, w: bw, hgt: bh, opacity: card },
      h(Txt, { x: G.text[0], y: G.text[1], size: L.fs(12.5), color: C.muted, weight: 500 }, "Brief fine-tune"),
      G.lines.map((t, i) => h(Txt, { key: i, x: G.text[0], y: G.text[2] + i * lh, size: L.fs(14), color: C.text }, t))));
}

const STEPS = {
  inputs: StepInputs, forward: StepForward, activations: StepActivations, distributions: StepDistributions, jm: StepJM,
  matrix: StepMatrix, space: StepSpace, mss: StepMSS, select: StepSelect, prune: StepPrune,
};

// ---------- player shell ----------
function useLayout(cardRef, figRef) {
  const [state, setState] = useState({ mode: "wide", scale: 1 });
  useLayoutEffect(() => {
    const card = cardRef.current, fig = figRef.current;
    if (!card || !fig) return undefined;
    const measure = () => {
      const cw = card.getBoundingClientRect().width;
      const mode = cw < NARROW_BELOW ? "narrow" : "wide";
      const cs = getComputedStyle(fig);
      const inner = fig.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      const figW = mode === "narrow" ? Math.min(inner, NARROW_MAX_FIG) : inner;
      const scale = Math.max(0.3, figW / VB[mode].w);
      setState((s) => (s.mode === mode && Math.abs(s.scale - scale) < 0.005 ? s : { mode, scale }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(card);
    return () => ro.disconnect();
  }, [cardRef, figRef]);
  return state;
}

function Icon({ name }) {
  const svg = (...kids) => h("svg", {
    width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 1.5,
    strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true, focusable: "false",
  }, ...kids);
  if (name === "play") return svg(h("path", { d: "M5 3.2 L13 8 L5 12.8 Z", fill: "currentColor" }));
  if (name === "pause") return svg(h("path", { d: "M5.5 3.5v9M10.5 3.5v9" }));
  if (name === "prev") return svg(h("path", { d: "M12 3.5 L5.5 8 L12 12.5 Z", fill: "currentColor" }), h("path", { d: "M3.5 3.5v9" }));
  if (name === "next") return svg(h("path", { d: "M4 3.5 L10.5 8 L4 12.5 Z", fill: "currentColor" }), h("path", { d: "M12.5 3.5v9" }));
  return svg(h("path", { d: "M13 8a5 5 0 1 1-1.6-3.7" }), h("path", { d: "M13.2 2.4v2.9h-2.9" }));
}

function CtlButton({ label, onClick, children }) {
  return h("button", { type: "button", className: "ms-btn", onClick, "aria-label": label, title: label }, children);
}

const SPEEDS = [0.5, 1, 1.5];

// ---------- step player ----------
// Plays the steps in order while the card is on screen. With reduced motion nothing starts by itself and every
// step is shown complete; pressing Play (or Replay) opts in to the animation, like starting a video.
function usePlayer(acts, reduced, inView) {
  const [st, setSt] = useState({ index: 0, progress: 0, playing: !reduced, optIn: false, ended: false });
  const [speed, setSpeed] = useState(1);
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const reducedRef = useRef(reduced);
  reducedRef.current = reduced;
  const animate = !reduced || st.optIn;
  const live = st.playing && animate;

  useEffect(() => {
    if (!live || !inView) return undefined;
    let raf = 0;
    let last = 0;
    const tick = (now) => {
      // Clamp the step so a backgrounded tab does not jump ahead when it becomes visible again.
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now;
      if (dt > 0) {
        setSt((s) => {
          if (!s.playing) return s;
          const progress = s.progress + (dt * speedRef.current) / acts[s.index].dwell;
          if (progress < 1) return { ...s, progress };
          if (s.index < acts.length - 1) return { ...s, index: s.index + 1, progress: 0 };
          return { ...s, progress: 1, playing: false, ended: true };
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [live, inView, acts]);

  // While playing, a newly chosen step animates from its start; while paused, it is shown complete.
  const move = useCallback((to) => setSt((s) => {
    const index = Math.max(0, Math.min(acts.length - 1, to(s.index)));
    if (index === s.index) return s;
    const moving = s.playing && (!reducedRef.current || s.optIn);
    return { ...s, index, progress: moving ? 0 : 1, ended: false };
  }), [acts.length]);

  return {
    index: st.index, act: acts[st.index], progress: animate ? st.progress : 1, playing: live, ended: st.ended, speed, setSpeed,
    goTo: (i) => move(() => i),
    next: () => move((i) => i + 1),
    prev: () => move((i) => i - 1),
    toggle: () => setSt((s) => {
      const anim = !reducedRef.current || s.optIn;
      if (s.playing && anim) return { ...s, playing: false };
      if (s.ended) return { ...s, index: 0, progress: 0, playing: true, optIn: true, ended: false };
      // A step that is shown complete (or was never animated) plays again from its start; otherwise resume.
      return { ...s, progress: !anim || s.progress >= 1 ? 0 : s.progress, playing: true, optIn: true };
    }),
    replay: () => setSt((s) => ({ ...s, progress: 0, playing: true, optIn: true, ended: false })),
  };
}

export function MethodStage() {
  const cardRef = useRef(null);
  const figRef = useRef(null);
  const tabRefs = useRef([]);
  const reduced = useReducedMotion();
  const inView = useInView(cardRef, "-10% 0px -10% 0px");
  const player = usePlayer(ACTS, reduced, inView);
  const { mode, scale } = useLayout(cardRef, figRef);
  const L = useMemo(() => {
    const narrow = mode === "narrow";
    // Below this scale (screens under ~360px) figure text shrinks with the figure instead of crowding it.
    const fscale = narrow ? Math.max(scale, 0.68) : scale;
    return { mode, narrow, W: VB[mode].w, H: VB[mode].h, scale, fscale, fs: (px) => +(px / fscale).toFixed(2) };
  }, [mode, scale]);

  const [leaving, setLeaving] = useState(null);
  const shown = useRef(player.index);
  useEffect(() => {
    if (shown.current === player.index) return undefined;
    const prev = ACTS[shown.current];
    shown.current = player.index;
    if (reduced || !prev) return undefined;
    setLeaving(prev.id);
    const t = setTimeout(() => setLeaving(null), 460);
    return () => clearTimeout(t);
  }, [player.index, reduced]);

  const act = player.act;
  const Step = STEPS[act.id];
  const Leaving = leaving && leaving !== act.id ? STEPS[leaving] : null;
  const p = player.progress;

  const onTabKey = (e) => {
    const last = ACTS.length - 1;
    const focused = tabRefs.current.indexOf(document.activeElement);
    const from = focused >= 0 ? focused : player.index;
    const to = { ArrowRight: Math.min(last, from + 1), ArrowLeft: Math.max(0, from - 1), Home: 0, End: last }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    player.goTo(to);
    tabRefs.current[to]?.focus();
  };

  // Left/Right arrows step through the walkthrough while it is on screen. Space is left to the page (scrolling).
  const playerRef = useRef(player);
  playerRef.current = player;
  useEffect(() => {
    if (!inView) return undefined;
    const onKey = (e) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const el = e.target;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
      e.preventDefault();
      if (e.key === "ArrowRight") playerRef.current.next();
      else playerRef.current.prev();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inView]);

  const tabs = h("div", { className: "ms-tabs", role: "tablist", "aria-label": "Steps of the method", onKeyDown: onTabKey },
    ACTS.map((a, i) => {
      const on = i === player.index;
      return h("button", {
        key: a.id, ref: (el) => { tabRefs.current[i] = el; }, type: "button", role: "tab", id: `ms-tab-${a.id}`,
        "aria-selected": on, "aria-controls": "ms-panel", "aria-label": `Step ${i + 1}: ${a.label}`, tabIndex: on ? 0 : -1,
        title: a.title, className: `ms-tab${i < player.index ? " is-done" : ""}`, onClick: () => player.goTo(i),
      },
      h("span", { className: "ms-tab-num", "aria-hidden": true }, i + 1),
      h("span", { className: "ms-tab-label", "aria-hidden": true }, a.label),
      on ? h("span", { className: "ms-tab-bar", style: { transform: `scaleX(${p})` }, "aria-hidden": true }) : null);
    }));

  const figure = h("div", { ref: figRef, className: "ms-figure", id: "ms-panel", role: "tabpanel", "aria-labelledby": `ms-tab-${act.id}` },
    h("svg", { viewBox: `0 0 ${L.W} ${L.H}`, role: "img", "aria-label": `${act.title}. ${act.caption}` },
      h(Defs, null),
      Leaving ? h("g", { key: `out-${leaving}`, className: "ms-fade-out" }, h(Leaving, { p: 1, L })) : null,
      h("g", { key: act.id, className: Leaving ? "ms-fade-in" : undefined }, h(Step, { p, L }))));

  const texts = h("div", { className: "ms-texts" }, ACTS.map((a, i) => {
    const on = i === player.index, eq = EQ[a.id];
    return h("div", { key: a.id, className: `ms-text${on ? " is-active" : ""}`, "aria-hidden": on ? undefined : true },
      h("div", { className: "ms-words" },
        h("div", { className: "ms-count" }, `Step ${i + 1} of ${ACTS.length}`),
        h("h3", { className: "ms-title" }, a.title),
        h("p", { className: "ms-caption" }, a.caption)),
      eq ? h("div", { className: "ms-eq" },
        h("div", { className: "ms-eq-math", dangerouslySetInnerHTML: { __html: L.narrow ? eq.narrow || eq.wide : eq.wide } }),
        h("div", { className: "ms-eq-ref" }, eq.ref)) : null);
  }));

  const controls = h("div", { className: "ms-controls" },
    h("div", { className: "ms-btns" },
      h(CtlButton, { label: "Previous step", onClick: player.prev }, h(Icon, { name: "prev" })),
      h(CtlButton, { label: player.playing ? "Pause" : player.ended ? "Play again from step 1" : "Play", onClick: player.toggle }, h(Icon, { name: player.playing ? "pause" : "play" })),
      h(CtlButton, { label: "Next step", onClick: player.next }, h(Icon, { name: "next" })),
      h(CtlButton, { label: "Replay this step", onClick: player.replay }, h(Icon, { name: "replay" }))),
    h("div", { className: "ms-speed", role: "group", "aria-label": "Playback speed" }, SPEEDS.map((s) => h("button", {
      key: s, type: "button", onClick: () => player.setSpeed(s), "aria-pressed": player.speed === s,
    }, `${s}×`))));

  return h("div", { ref: cardRef, className: "ms-card", "data-mode": mode },
    tabs, figure, controls,
    h("div", { className: "ms-body" }, texts,
      reduced ? h("p", { className: "ms-note" }, "Reduced motion is on, so the walkthrough does not start by itself and each step is shown complete. Press play to watch it animate.") : null));
}
