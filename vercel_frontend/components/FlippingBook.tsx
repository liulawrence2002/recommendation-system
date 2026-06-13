/* A pure-CSS 3D book whose pages continuously riffle — the literary centrepiece,
   carried over from the Streamlit app. With `ambient`, it gains a warm glow halo
   and pages that lift off and drift upward, for a cinematic hero moment. */
export default function FlippingBook({
  leaves = 4,
  ambient = false,
}: {
  leaves?: number;
  ambient?: boolean;
}) {
  return (
    <div className={`book-stage${ambient ? " book-stage--ambient" : ""}`} aria-hidden="true">
      {ambient && <div className="book-glow" />}
      {ambient && (
        <div className="book-embers">
          {Array.from({ length: 6 }, (_, i) => (
            <span key={i} className={`ember ember-${i + 1}`} />
          ))}
        </div>
      )}
      <div className="book">
        <div className="book-3d">
          <div className="book-cover" />
          <div className="page-static page-left" />
          <div className="page-static page-right" />
          {Array.from({ length: leaves }, (_, i) => (
            <div key={i} className={`leaf leaf-${i + 1}`} />
          ))}
          <div className="spine" />
        </div>
      </div>
    </div>
  );
}
