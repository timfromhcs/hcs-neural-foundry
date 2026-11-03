/**
 * Presentation primitives.
 *
 * Accessibility rules applied here rather than per page:
 *  - status is never encoded by colour alone (badges carry text);
 *  - charts expose a text summary via <title>/<desc> and an aria-label;
 *  - every interactive element is a real button/input, keyboard reachable by default.
 */

import { useId, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";

import { useT } from "../i18n";
import type { Capability, Maturity } from "../lib/types";

export function Card({
  title,
  subtitle,
  actions,
  children,
  tight,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  tight?: boolean;
}) {
  const headingId = useId();
  return (
    <section className={`card${tight ? " tight" : ""}`} aria-labelledby={title ? headingId : undefined}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h3 id={headingId}>{title}</h3>}
            {subtitle && <small className="faint">{subtitle}</small>}
          </div>
          {actions && <div className="row">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function PageHead({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <header className="page-head">
      <div>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {actions && <div className="row">{actions}</div>}
    </header>
  );
}

export function Badge({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`badge ${className}`}>{children}</span>;
}

export function MaturityBadge({ maturity }: { maturity: Maturity }) {
  const t = useT();
  const label =
    maturity === "stable" ? t("maturity.stable") : maturity === "experimental" ? t("maturity.experimental") : t("maturity.unsupported");
  return <span className={`badge ${maturity}`}>{label}</span>;
}

export function CapabilityBadge({ value }: { value: Capability }) {
  const t = useT();
  const map: Record<Capability, { label: string; cls: string }> = {
    full: { label: t("parity.full"), cls: "stable" },
    limited: { label: t("parity.limited"), cls: "experimental" },
    experimental: { label: t("parity.experimental"), cls: "experimental" },
    unavailable: { label: t("parity.unavailable"), cls: "unsupported" },
  };
  const entry = map[value];
  return <span className={`badge ${entry.cls}`}>{entry.label}</span>;
}

export function StatusPill({ tone, children }: { tone: "ok" | "warn" | "bad" | "neutral"; children: ReactNode }) {
  return (
    <span className={`badge ${tone === "neutral" ? "neutral" : ""} ${tone === "ok" ? "stable" : tone === "warn" ? "experimental" : tone === "bad" ? "unsupported" : ""}`}>
      {children}
    </span>
  );
}

export function Alert({ tone = "info", title, children }: { tone?: "info" | "ok" | "warn" | "bad"; title?: string; children: ReactNode }) {
  const icon = tone === "ok" ? "✓" : tone === "warn" ? "!" : tone === "bad" ? "×" : "i";
  return (
    <div className={`alert ${tone === "info" ? "" : tone}`} role={tone === "bad" ? "alert" : "note"}>
      <span className="alert-icon" aria-hidden="true">
        {icon}
      </span>
      <div>
        {title && <strong>{title}</strong>}
        {title ? <div>{children}</div> : children}
      </div>
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
  htmlFor,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  htmlFor?: string;
}) {
  const generated = useId();
  const id = htmlFor ?? generated;
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
      {hint && (
        <span className="help" id={`${id}-hint`}>
          {hint}
        </span>
      )}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="kpi">
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      {hint && <span className="kpi-hint">{hint}</span>}
    </div>
  );
}

export function Meter({ value, max = 1, label }: { value: number; max?: number; label: string }) {
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return (
    <div>
      <div className="meter" role="meter" aria-valuenow={Math.round(ratio * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
        <i style={{ width: `${ratio * 100}%` }} />
      </div>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function CopyButton({ value, label }: { value: string; label?: string }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="btn small ghost"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1400);
        } catch {
          setCopied(false);
        }
      }}
      aria-live="polite"
    >
      {copied ? t("common.copied") : (label ?? t("common.copy"))}
    </button>
  );
}

// -------------------------------------------------------------------------------------------------
// Charts
// -------------------------------------------------------------------------------------------------

export function LineChart({
  points,
  labels,
  height = 180,
  title,
  unit,
}: {
  points: { x: number; y: number }[];
  labels: { x: string; y: string };
  height?: number;
  title: string;
  unit?: string;
}) {
  const width = 640;
  const padding = { left: 44, right: 12, top: 12, bottom: 26 };
  const clean = points.filter((point) => Number.isFinite(point.y));
  if (clean.length < 2) {
    return <div className="empty">—</div>;
  }
  const xs = clean.map((point) => point.x);
  const ys = clean.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const scaleX = (value: number) => padding.left + ((value - minX) / spanX) * (width - padding.left - padding.right);
  const scaleY = (value: number) =>
    height - padding.bottom - ((value - minY) / spanY) * (height - padding.top - padding.bottom);
  const path = clean.map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.x).toFixed(2)} ${scaleY(point.y).toFixed(2)}`).join(" ");
  const area = `${path} L ${scaleX(maxX).toFixed(2)} ${height - padding.bottom} L ${scaleX(minX).toFixed(2)} ${height - padding.bottom} Z`;
  const last = clean[clean.length - 1]!;
  const summary = `${title}: ${clean.length} points, from ${labels.y} ${minY.toFixed(4)} to ${last.y.toFixed(4)}${unit ? ` ${unit}` : ""} at ${labels.x} ${last.x}.`;

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={summary}>
      <title>{title}</title>
      <desc>{summary}</desc>
      <g className="chart-grid">
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
          <line key={ratio} x1={padding.left} x2={width - padding.right} y1={scaleY(minY + ratio * spanY)} y2={scaleY(minY + ratio * spanY)} />
        ))}
      </g>
      <path className="chart-area" d={area} />
      <path className="chart-line" d={path} />
      <circle cx={scaleX(last.x)} cy={scaleY(last.y)} r={3.5} fill="var(--accent)" />
      <g className="chart-axis">
        <text x={4} y={scaleY(maxY) + 4}>{maxY.toFixed(3)}</text>
        <text x={4} y={scaleY(minY) + 4}>{minY.toFixed(3)}</text>
        <text x={padding.left} y={height - 8}>{labels.x} {minX.toFixed(0)}</text>
        <text x={width - padding.right} y={height - 8} textAnchor="end">{labels.x} {maxX.toFixed(0)}</text>
      </g>
    </svg>
  );
}

export function BarChart({
  values,
  title,
  labels,
}: {
  values: number[];
  title: string;
  labels: { x: string; y: string };
}) {
  const width = 640;
  const height = 160;
  const max = Math.max(...values, 1);
  const barWidth = width / Math.max(values.length, 1);
  const summary = `${title}: ${values.length} ${labels.x} bins, maximum ${max}.`;
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={summary}>
      <title>{title}</title>
      <desc>{summary}</desc>
      {values.map((value, index) => {
        const barHeight = (value / max) * (height - 24);
        return (
          <rect
            key={index}
            className="chart-bar"
            x={index * barWidth + 1}
            y={height - 20 - barHeight}
            width={Math.max(1, barWidth - 2)}
            height={barHeight}
            opacity={0.55 + 0.45 * (value / max)}
          />
        );
      })}
      <g className="chart-axis">
        <text x={2} y={10}>{labels.y}</text>
        <text x={width - 2} y={height - 4} textAnchor="end">{labels.x}</text>
      </g>
    </svg>
  );
}

export function Heatmap({
  values,
  rows,
  cols,
  title,
}: {
  values: number[];
  rows: number;
  cols: number;
  title: string;
}) {
  const finite = values.filter((value) => Number.isFinite(value));
  const maxAbs = Math.max(...finite.map((value) => Math.abs(value)), 1e-9);
  const summary = `${title}: ${rows} by ${cols} matrix, max absolute value ${maxAbs.toFixed(4)}.`;
  return (
    <div className="heatmap" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }} role="img" aria-label={summary}>
      {values.slice(0, rows * cols).map((value, index) => {
        const ratio = Math.min(1, Math.abs(value) / maxAbs);
        const positive = value >= 0;
        const color = positive
          ? `color-mix(in srgb, var(--accent) ${Math.round(ratio * 90)}%, transparent)`
          : `color-mix(in srgb, var(--bad) ${Math.round(ratio * 90)}%, transparent)`;
        return <span key={index} style={{ background: color }} title={value.toFixed(5)} />;
      })}
    </div>
  );
}

export function Tabs<T extends string>({
  items,
  value,
  onChange,
  ariaLabel,
}: {
  items: { id: T; label: string }[];
  value: T;
  onChange: (id: T) => void;
  ariaLabel: string;
}) {
  return (
    <div className="tabs" role="tablist" aria-label={ariaLabel}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          id={`tab-${item.id}`}
          className="tab"
          aria-selected={value === item.id}
          aria-controls={`panel-${item.id}`}
          onClick={() => onChange(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  id,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
  id?: string;
}) {
  return (
    <select id={id} value={value} onChange={(event: ChangeEvent<HTMLSelectElement>) => onChange(event.target.value as T)}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}
