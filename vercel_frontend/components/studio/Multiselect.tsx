"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

/** A compact searchable multi-select. Handles the ~5,800-author list (caps the
    rendered matches) and the small decade list alike. */
export default function Multiselect({
  label,
  options,
  selected,
  onChange,
  placeholder = "Search…",
  emptyText = "No matches",
  maxVisible = 60,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  emptyText?: string;
  maxVisible?: number;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const inputId = useId();

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    const sel = new Set(selected);
    const pool = q ? options.filter((o) => o.toLowerCase().includes(q)) : options;
    return pool.filter((o) => !sel.has(o)).slice(0, maxVisible);
  }, [query, options, selected, maxVisible]);

  function toggle(value: string) {
    if (selected.includes(value)) onChange(selected.filter((s) => s !== value));
    else onChange([...selected, value]);
    setQuery("");
  }

  return (
    <div className="ms" ref={ref}>
      <label className="field-label" htmlFor={inputId}>
        {label}
      </label>
      <div
        className="ms-control"
        role="combobox"
        aria-expanded={open}
        aria-controls={`${inputId}-menu`}
        onClick={() => setOpen(true)}
      >
        {selected.map((s) => (
          <span className="ms-chip" key={s}>
            {s}
            <button
              type="button"
              className="ms-chip-x"
              aria-label={`Remove ${s}`}
              onClick={(e) => {
                e.stopPropagation();
                onChange(selected.filter((x) => x !== s));
              }}
            >
              ×
            </button>
          </span>
        ))}
        <input
          id={inputId}
          className="ms-input"
          value={query}
          aria-label={label}
          placeholder={selected.length ? "" : placeholder}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === "Escape") setOpen(false);
            else if (e.key === "Backspace" && query === "" && selected.length) {
              onChange(selected.slice(0, -1));
            }
          }}
        />
      </div>
      {open && (
        <div className="ms-menu" role="listbox" id={`${inputId}-menu`}>
          {matches.length === 0 ? (
            <div className="ms-empty">{emptyText}</div>
          ) : (
            matches.map((o) => (
              <button
                type="button"
                className="ms-option"
                key={o}
                role="option"
                aria-selected={false}
                onClick={() => toggle(o)}
              >
                {o}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
