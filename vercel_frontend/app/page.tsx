import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import HowItWorks from "@/components/HowItWorks";
import Pipeline from "@/components/Pipeline";
import Features from "@/components/Features";
import CtaBand from "@/components/CtaBand";
import Footer from "@/components/Footer";
import { getStats } from "@/lib/serverData";

export default function Home() {
  const { headline } = getStats();
  return (
    <>
      <Nav variant="landing" />
      <main className="page">
        <Hero users={headline.users} books={headline.books} ratings={headline.ratings} />
        <HowItWorks />
        <Pipeline />
        <Features />
        <CtaBand />
      </main>
      <Footer />
    </>
  );
}
