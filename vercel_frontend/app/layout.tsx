import type { Metadata, Viewport } from "next";
import { Fraunces, Inter } from "next/font/google";
import MotionProvider from "@/components/MotionProvider";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-serif",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "BookRec — a conversational book recommender",
  description:
    "Collaborative-filtering candidates, re-ranked by a conversational RAG/DAG that reads your mood, asks when it's unsure, and explains every pick. Built on the Goodreads dataset.",
  keywords: [
    "book recommender",
    "collaborative filtering",
    "RAG",
    "LLM re-ranking",
    "Goodreads",
  ],
  openGraph: {
    title: "BookRec — a conversational book recommender",
    description:
      "Collaborative filtering meets a conversational RAG/DAG re-ranker. Warm, literary, explainable.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#f4ecdd",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${fraunces.variable} ${inter.variable}`}>
      <body>
        <div className="fluid-bg" aria-hidden="true">
          <div className="orb orb-1" />
          <div className="orb orb-2" />
          <div className="orb orb-3" />
        </div>
        <MotionProvider>{children}</MotionProvider>
      </body>
    </html>
  );
}
