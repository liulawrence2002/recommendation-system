/* Ambient, slowly-drifting SVG art layered behind the content (inside .fluid-bg).
   - a node constellation: the collaborative-filtering / recommendation graph
   - a spiral of pages: the literary motif
   - a soft amber light-leak
   Pure SVG + CSS animation; parallax is applied by the .par wrappers reading the
   CSS offset variables set by <ParallaxController />. Decorative, aria-hidden. */

const NODES = [
  { x: 56, y: 54, r: 4 },
  { x: 138, y: 36, r: 2.6 },
  { x: 226, y: 74, r: 3.4 },
  { x: 300, y: 44, r: 2.4 },
  { x: 96, y: 128, r: 3 },
  { x: 186, y: 150, r: 5 },
  { x: 278, y: 162, r: 3 },
  { x: 46, y: 206, r: 2.6 },
  { x: 150, y: 232, r: 3.4 },
  { x: 244, y: 238, r: 2.6 },
  { x: 322, y: 206, r: 3.2 },
];

const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [0, 4], [1, 5], [4, 5], [5, 6], [2, 6],
  [4, 7], [5, 8], [7, 8], [8, 9], [6, 10], [9, 10], [3, 6], [5, 9],
];

function spiralPath(): string {
  const pts: string[] = [];
  for (let t = 0; t <= 7 * Math.PI; t += 0.18) {
    const r = 6 + 6.4 * t;
    pts.push(`${(150 + r * Math.cos(t)).toFixed(1)},${(150 + r * Math.sin(t)).toFixed(1)}`);
  }
  return `M ${pts.join(" L ")}`;
}

export default function BackgroundArt() {
  const spiral = spiralPath();
  return (
    <div className="bg-art" aria-hidden="true">
      <div className="par par-slow">
        <div className="bg-constellation">
          <svg viewBox="0 0 360 280" fill="none">
            <g stroke="#a35421" strokeWidth="1" opacity="0.5">
              {EDGES.map(([a, b], i) => (
                <line key={i} x1={NODES[a].x} y1={NODES[a].y} x2={NODES[b].x} y2={NODES[b].y} />
              ))}
            </g>
            <g fill="#7c4a23">
              {NODES.map((n, i) => (
                <circle key={i} cx={n.x} cy={n.y} r={n.r} />
              ))}
            </g>
            {/* one highlighted "you" node */}
            <circle cx={NODES[5].x} cy={NODES[5].y} r="8" fill="none" stroke="#c9802f" strokeWidth="1.4" />
          </svg>
        </div>
      </div>

      <div className="par par-fast">
        <div className="bg-spiral">
          <svg viewBox="0 0 300 300" fill="none">
            <path d={spiral} stroke="#a35421" strokeWidth="1.4" strokeLinecap="round" strokeDasharray="1 9" />
            <path d={spiral} stroke="#7c4a23" strokeWidth="1" opacity="0.4" />
          </svg>
        </div>
      </div>

      <div className="light-leak" />
    </div>
  );
}
