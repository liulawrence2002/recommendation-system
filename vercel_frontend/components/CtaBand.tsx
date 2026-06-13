import Link from "next/link";
import Reveal from "./Reveal";

export default function CtaBand() {
  return (
    <section className="section">
      <div className="container">
        <Reveal>
          <div className="cta-band">
            <div className="cta-glow" aria-hidden="true" />
            <h2>Tell it what you&apos;re in the mood for.</h2>
            <p>
              Build a candidate shortlist, then refine it in conversation — steer the tone,
              pace, and setting, and watch the reasoning unfold for every pick.
            </p>
            <Link className="btn btn-primary" href="/studio">
              Open the studio →
            </Link>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
