/* A pure-CSS 3D book whose pages continuously riffle — the literary centrepiece,
   carried over from the Streamlit app's hero animation. */
export default function FlippingBook({ leaves = 4 }: { leaves?: number }) {
  return (
    <div className="book-stage" aria-hidden="true">
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
