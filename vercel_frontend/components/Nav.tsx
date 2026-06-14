import Link from "next/link";

type NavLink = { label: string; href: string };

const LANDING_LINKS: NavLink[] = [
  { label: "How it works", href: "#how" },
  { label: "The pipeline", href: "#pipeline" },
  { label: "Why it matters", href: "#why" },
];

const STUDIO_LINKS: NavLink[] = [
  { label: "Filter", href: "#filter" },
  { label: "The shelf", href: "#shelf" },
  { label: "Refine", href: "#chat" },
];

export default function Nav({
  variant = "landing",
  status,
}: {
  variant?: "landing" | "studio";
  status?: { label: string; idle?: boolean };
}) {
  const links = variant === "studio" ? STUDIO_LINKS : LANDING_LINKS;
  const cta =
    variant === "studio"
      ? { label: "Personalize", href: "#chat" }
      : { label: "Open the studio", href: "/studio" };

  return (
    <nav className="nav" aria-label="BookRec">
      <div className="nav-inner">
        <Link className="brand" href="/">
          <span className="brand-mark">B</span>
          <span className="brand-words">
            <span className="brand-name">BookRec</span>
            <span className="brand-tag">Conversational recommendations</span>
          </span>
        </Link>
        <div className="nav-links">
          {links.map((l) => (
            <a key={l.href} className="nav-pill" href={l.href}>
              {l.label}
            </a>
          ))}
        </div>
        {status && (
          <span className="nav-status">
            <span className={`status-dot${status.idle ? " idle" : ""}`} />
            {status.label}
          </span>
        )}
        <Link className="nav-cta" href={cta.href}>
          {cta.label}
        </Link>
      </div>
    </nav>
  );
}
