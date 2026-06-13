import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Book covers are remote Goodreads URLs rendered with plain <img> tags, so no
  // next/image remote-pattern allow-list is required. Nothing else to configure —
  // this is a standard App Router project that deploys to Vercel as-is.
};

export default nextConfig;
