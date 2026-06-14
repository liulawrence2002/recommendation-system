import { redirect } from "next/navigation";

// Streamlined: the whole experience is the cinematic studio flow
// (video cold-open → filter + shelf → conversational refine). The marketing
// landing is hidden for now; the root sends visitors straight into the flow.
export default function Home() {
  redirect("/studio");
}
