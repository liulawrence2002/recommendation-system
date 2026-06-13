import Reveal from "./Reveal";

const FEATURES = [
  {
    icon: "◆",
    title: "Grounded, never invented",
    copy: "The LLM only ever re-ranks the collaborative-filtering candidates. Every stage re-validates book ids against the shortlist — no hallucinated titles, ever.",
  },
  {
    icon: "✶",
    title: "Explainable by design",
    copy: "Each pick comes with one neutral sentence about the book and one on why it earns its rank for you. The full reasoning trace is one tap away.",
  },
  {
    icon: "❋",
    title: "Always answers",
    copy: "No API key? The whole pipeline runs on a transparent, deterministic heuristic. Add a Gemini key and the same DAG upgrades to live reasoning.",
  },
];

export default function Features() {
  return (
    <section className="section" id="why">
      <div className="container">
        <Reveal className="section-head">
          <div className="eyebrow">Why it matters</div>
          <h2 className="section-title">
            Collaborative filtering scales. Language makes it personal.
          </h2>
          <p className="section-copy">
            CF is cheap to serve and driven purely by behaviour, but it can&apos;t read
            &ldquo;a cozy mystery, nothing gory.&rdquo; A language layer captures the intent
            ratings can&apos;t — with the guardrails a business actually needs.
          </p>
        </Reveal>
        <div className="feature-grid">
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} delay={i * 0.08}>
              <div className="feature">
                <div className="feature-icon">{f.icon}</div>
                <div className="feature-title">{f.title}</div>
                <div className="feature-copy">{f.copy}</div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
