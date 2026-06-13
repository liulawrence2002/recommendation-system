export default function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-inner">
        <span>
          BookRec — collaborative filtering + a conversational RAG/DAG re-ranker, on the
          Goodreads dataset.
        </span>
        <span>
          Candidates: UBCF / IBCF / SVD · Re-ranking: Gemini with a deterministic fallback.
        </span>
      </div>
    </footer>
  );
}
