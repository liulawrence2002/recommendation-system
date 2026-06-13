import { ImageResponse } from "next/og";

// Cinematic 1200x630 social share card, rendered to PNG by Satori. No external
// font fetch (robust on build + edge); the warm gradient, the open-book motif,
// and a strong type hierarchy carry the literary, premium feel.

export const alt = "BookRec — find your next book by talking it through";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const BOOK_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 360">
  <defs>
    <radialGradient id="glow" cx="50%" cy="46%" r="56%">
      <stop offset="0" stop-color="#eab66f" stop-opacity="0.6"/>
      <stop offset="1" stop-color="#eab66f" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="cover" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#8a5326"/><stop offset="1" stop-color="#5d3618"/>
    </linearGradient>
    <linearGradient id="pageL" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fdf9ef"/><stop offset="1" stop-color="#eaddc0"/>
    </linearGradient>
    <linearGradient id="pageR" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fffdf8"/><stop offset="1" stop-color="#f1e5cd"/>
    </linearGradient>
    <linearGradient id="amber" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#e6ab5f"/><stop offset="1" stop-color="#a35421"/>
    </linearGradient>
  </defs>
  <circle cx="220" cy="190" r="190" fill="url(#glow)"/>
  <g transform="translate(0,10)">
    <path d="M220 128 C 158 99 82 103 40 128 L 40 262 C 82 237 158 241 220 270 Z" fill="url(#cover)"/>
    <path d="M220 128 C 282 99 358 103 400 128 L 400 262 C 358 237 282 241 220 270 Z" fill="url(#cover)"/>
  </g>
  <path d="M220 120 C 160 93 86 97 46 121 L 46 252 C 86 228 160 232 220 260 Z" fill="url(#pageL)"/>
  <path d="M220 120 C 280 93 354 97 394 121 L 394 252 C 354 228 280 232 220 260 Z" fill="url(#pageR)"/>
  <g stroke="#d4be93" stroke-width="2" stroke-linecap="round" fill="none" opacity="0.75">
    <path d="M72 142 C 110 130 150 132 196 150"/>
    <path d="M72 166 C 110 154 150 156 196 174"/>
    <path d="M72 190 C 110 178 150 180 196 198"/>
    <path d="M72 214 C 110 202 150 204 196 222"/>
    <path d="M244 150 C 290 132 330 130 368 142"/>
    <path d="M244 174 C 290 156 330 154 368 166"/>
    <path d="M244 198 C 290 180 330 178 368 190"/>
    <path d="M244 222 C 290 204 330 202 368 214"/>
  </g>
  <path d="M220 120 L 220 260" stroke="url(#amber)" stroke-width="3.5" stroke-linecap="round"/>
  <!-- pages lifting off -->
  <g opacity="0.9">
    <path d="M250 86 l 40 -10 l 8 30 l -40 10 z" fill="#fbf3e0" stroke="#e3d2ac" stroke-width="1.5" transform="rotate(-14 270 96)"/>
    <path d="M300 60 l 34 -6 l 6 26 l -34 6 z" fill="#fdf8ee" stroke="#e3d2ac" stroke-width="1.5" transform="rotate(-7 318 70)"/>
    <path d="M196 70 l 34 8 l -6 26 l -34 -8 z" fill="#fbf3e0" stroke="#e3d2ac" stroke-width="1.5" transform="rotate(10 212 84)"/>
  </g>
  <path d="M338 44 l 4.5 12 l 12 4.5 l -12 4.5 l -4.5 12 l -4.5 -12 l -12 -4.5 l 12 -4.5 z" fill="url(#amber)"/>
</svg>`;

const bookUri = `data:image/svg+xml;utf8,${encodeURIComponent(BOOK_SVG)}`;

export default function Image() {
  const ink = "#2a2118";
  const muted = "#6a5c45";
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          position: "relative",
          padding: "60px 80px",
          color: ink,
          backgroundColor: "#f4ecdd",
          backgroundImage:
            "radial-gradient(900px 520px at 18% -10%, rgba(214,158,86,0.40), rgba(214,158,86,0) 60%)," +
            "radial-gradient(760px 600px at 100% 120%, rgba(124,74,35,0.22), rgba(124,74,35,0) 62%)," +
            "linear-gradient(150deg, #f8f1e3 0%, #efe3cf 55%, #ead9bf 100%)",
          fontFamily: "sans-serif",
        }}
      >
        {/* left: copy */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            flex: 1,
            paddingRight: 24,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 18, marginBottom: 26 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 64,
                height: 64,
                borderRadius: 18,
                background: "linear-gradient(150deg, #7c4a23, #2a2118)",
                color: "#fbf6ec",
                fontSize: 36,
                fontWeight: 700,
                boxShadow: "0 10px 24px rgba(42,33,24,0.3)",
              }}
            >
              B
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <div style={{ fontSize: 30, fontWeight: 700, letterSpacing: -0.5 }}>BookRec</div>
              <div style={{ fontSize: 18, color: muted }}>Conversational book recommender</div>
            </div>
          </div>

          <div
            style={{
              fontSize: 56,
              fontWeight: 600,
              lineHeight: 1.06,
              letterSpacing: -1.2,
              maxWidth: 600,
            }}
          >
            Find your next book by talking it through.
          </div>

          <div
            style={{
              fontSize: 23,
              lineHeight: 1.45,
              color: muted,
              marginTop: 22,
              maxWidth: 560,
            }}
          >
            Collaborative filtering, re-ranked by a conversational RAG that explains every pick.
          </div>

          <div style={{ display: "flex", gap: 14, marginTop: 32 }}>
            {["9,964 books", "164,728 ratings", "3-stage reasoning DAG"].map((c) => (
              <div
                key={c}
                style={{
                  display: "flex",
                  fontSize: 20,
                  fontWeight: 600,
                  color: "#a35421",
                  background: "rgba(243,230,210,0.85)",
                  border: "1px solid #e7d2b4",
                  borderRadius: 999,
                  padding: "10px 18px",
                }}
              >
                {c}
              </div>
            ))}
          </div>
        </div>

        {/* right: book motif */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 440 }}>
          <img src={bookUri} width={440} height={360} alt="" />
        </div>
      </div>
    ),
    { ...size }
  );
}
