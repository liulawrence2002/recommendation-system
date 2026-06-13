import Reveal from "./Reveal";

const FLOW = ["Your message", "Intent", "Clarify?", "Score", "Re-rank", "Shortlist"];

const TRACE = [
  {
    no: 1,
    name: "Understand your request",
    badge: "AI",
    live: true,
    note: "We read your message and pulled out the details that matter.",
    chips: ["dark", "atmospheric", "mystery", "slow pace"],
    avoid: ["gore"],
  },
  {
    no: 2,
    name: "Check we have enough",
    badge: "Smart rules",
    live: false,
    note: "Clear enough to act on — straight to picking books, no question needed.",
  },
  {
    no: 3,
    name: "Rate every book",
    badge: "AI",
    live: true,
    note: "We rated all 10 candidates on how well they fit, then kept the strongest.",
  },
  {
    no: 4,
    name: "Pick the final list",
    badge: "AI",
    live: true,
    note: "We ordered the best matches and wrote a short reason for each.",
  },
];

export default function Pipeline() {
  return (
    <section className="section" id="pipeline">
      <div className="container">
        <Reveal className="section-head">
          <div className="eyebrow">The pipeline</div>
          <h2 className="section-title">A reasoning DAG you can read.</h2>
          <p className="section-copy">
            Every recommendation carries its work. The same sequential pipeline runs with a
            live LLM or a transparent rule-based fallback — so it always answers, and always
            shows its steps.
          </p>
        </Reveal>

        <Reveal>
          <div className="flow">
            {FLOW.map((node, i) => (
              <div className="flow-item" key={node}>
                <span className="flow-node">{node}</span>
                {i < FLOW.length - 1 && <span className="flow-arrow">→</span>}
              </div>
            ))}
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="card card-pad trace-demo">
            <div className="trace-stages">
              {TRACE.map((s) => (
                <div className="trace-stage" key={s.no}>
                  <div className="trace-stage-head">
                    <span className="trace-step-no">{s.no}</span>
                    <span className="trace-stage-name">{s.name}</span>
                    <span className={`stage-badge ${s.live ? "live" : "heuristic"}`}>
                      <span className="status-dot" />
                      {s.badge}
                    </span>
                  </div>
                  <div className="trace-note">{s.note}</div>
                  {(s.chips || s.avoid) && (
                    <div className="trace-chips" style={{ marginTop: "0.5rem" }}>
                      {s.chips?.map((c) => (
                        <span className="trace-chip" key={c}>
                          {c}
                        </span>
                      ))}
                      {s.avoid?.map((c) => (
                        <span className="trace-chip avoid" key={c}>
                          avoid {c}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
