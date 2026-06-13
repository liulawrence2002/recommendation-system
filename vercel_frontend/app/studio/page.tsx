import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import StudioApp from "@/components/studio/StudioApp";

// Read GEMINI_API_KEY at request time so the nav reflects the live mode.
export const dynamic = "force-dynamic";

export default function StudioPage() {
  const hasKey = Boolean(process.env.GEMINI_API_KEY);
  return (
    <>
      <Nav
        variant="studio"
        status={{ label: hasKey ? "Gemini ready" : "Heuristic mode", idle: !hasKey }}
      />
      <main className="page">
        <header className="studio-hero">
          <div className="container">
            <div className="eyebrow">BookRec studio</div>
            <h1 className="hero-title">Build the shortlist, then talk it into shape.</h1>
            <p className="hero-copy">
              Generate collaborative-filtering candidates, then refine them in a conversation that
              reads your intent, asks when it&apos;s unsure, and explains every pick.
            </p>
          </div>
        </header>
        <StudioApp />
      </main>
      <Footer />
    </>
  );
}
