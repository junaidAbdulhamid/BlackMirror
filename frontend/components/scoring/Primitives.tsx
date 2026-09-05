/**
 * Shared presentation primitives, matched to the TRIBE v2 demo's design
 * language: pure-black ground, 2px corners, wide-tracked uppercase labels,
 * muted slate body copy, and numerals in tabular figures so columns align.
 */

import type { ReactNode } from "react";

export function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`tv-label ${className}`}>{children}</div>;
}

export function Panel({
  title,
  aside,
  children,
  className = "",
}: {
  title?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`tv-panel ${className}`}>
      {title ? (
        <header className="flex items-baseline justify-between border-b border-[var(--line)] px-5 py-3">
          <Label>{title}</Label>
          {aside}
        </header>
      ) : null}
      <div className="p-5">{children}</div>
    </section>
  );
}

/**
 * A measured number. `raw` is always rendered; a score is secondary, because
 * normalization can make a negligible raw difference look decisive.
 */
export function Metric({
  value,
  caption,
  emphasis = false,
}: {
  value: string;
  caption?: string;
  emphasis?: boolean;
}) {
  return (
    <div>
      <div
        className={`tv-num tv-display ${emphasis ? "text-[2.5rem] leading-none" : "text-[1.125rem]"}`}
      >
        {value}
      </div>
      {caption ? <div className="tv-label mt-1.5">{caption}</div> : null}
    </div>
  );
}

/** A non-dismissible statement of a measured caveat. Never styled as an error. */
export function Notice({ children }: { children: ReactNode }) {
  return (
    <p className="border-l border-[var(--line-strong)] py-0.5 pl-3 text-[11px] leading-relaxed text-[var(--muted)]">
      {children}
    </p>
  );
}

export function Bar({ fraction, muted = false }: { fraction: number; muted?: boolean }) {
  const width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  return (
    <div className="h-[3px] w-full bg-[var(--line)]">
      <div
        className="h-full"
        style={{ width, background: muted ? "var(--muted)" : "var(--foreground)" }}
      />
    </div>
  );
}
