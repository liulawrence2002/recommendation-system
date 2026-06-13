import Reveal from "./Reveal";

const STEPS = [
  {
    title: "Shortlist with collaborative filtering",
    copy: "User-based CF (Pearson) surfaces the books that readers with your taste rated highly — a focused candidate pool, not the whole catalog.",
    tag: "Candidates",
  },
  {
    title: "Understand your request",
    copy: "Stage A distills your message into structured intent: mood, genres, pace, recency, and anything to steer away from.",
    tag: "Stage A · Intent",
  },
  {
    title: "Ask only when it helps",
    copy: "A clarify gate measures how much signal your request carries. Too thin? It asks one sharp question instead of guessing.",
    tag: "Gate",
  },
  {
    title: "Score, re-rank, explain",
    copy: "Stages B & C rate every candidate against your intent and order the best — with a one-line reason each, never inventing a title.",
    tag: "Stage B · C",
  },
];

export default function HowItWorks() {
  return (
    <section className="section" id="how">
      <div className="container">
        <Reveal className="section-head">
          <div className="eyebrow">How it works</div>
          <h2 className="section-title">From a sentence to a shortlist you can trust.</h2>
          <p className="section-copy">
            Two ideas, composed: collaborative filtering decides <em>which</em> books are
            worth considering; a conversational reasoning pipeline decides which of those
            fit <em>you</em>, right now — and tells you why.
          </p>
        </Reveal>
        <div className="steps">
          {STEPS.map((s, i) => (
            <Reveal key={s.title} delay={i * 0.08}>
              <div className="step">
                <div className="step-no">{i + 1}</div>
                <div className="step-title">{s.title}</div>
                <div className="step-copy">{s.copy}</div>
                <span className="step-tag">{s.tag}</span>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
