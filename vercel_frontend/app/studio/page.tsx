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
      <main className="page" id="top" style={{ paddingTop: "calc(var(--nav-h) + 2.5rem)" }}>
        <StudioApp />
      </main>
      <Footer />
    </>
  );
}
