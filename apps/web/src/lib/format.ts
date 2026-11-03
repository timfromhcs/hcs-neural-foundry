/** Locale-aware formatting helpers. Numbers and dates never use a hardcoded separator. */

export type Locale = "en" | "de" | "fr";

const LOCALE_TAG: Record<Locale, string> = { en: "en-US", de: "de-DE", fr: "fr-FR" };

export function formatNumber(value: number, locale: Locale, digits = 0): string {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(LOCALE_TAG[locale], {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatCompact(value: number, locale: Locale): string {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(LOCALE_TAG[locale], { notation: "compact", maximumFractionDigits: 2 }).format(value);
}

export function formatPercent(value: number, locale: Locale, digits = 1): string {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(LOCALE_TAG[locale], {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatBytes(bytes: number, locale: Locale): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  const units = ["B", "kB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${formatNumber(value, locale, value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

export function formatDate(iso: string, locale: Locale): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return new Intl.DateTimeFormat(LOCALE_TAG[locale], {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function formatDuration(ms: number, locale: Locale): string {
  if (!Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${formatNumber(ms, locale, 0)} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${formatNumber(seconds, locale, 1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${formatNumber(seconds % 60, locale, 0)} s`;
}

/** Short, stable hash display: first 12 hex characters plus an ellipsis. */
export function shortHash(hash: string, length = 12): string {
  return hash.length > length ? `${hash.slice(0, length)}…` : hash;
}

/**
 * Synchronous SHA-256 fallback used only when `crypto.subtle` is unavailable (insecure contexts).
 * Produces a deterministic content hash for the browser's own provenance records.
 */
export function weakHash(text: string): string {
  let h1 = 0x9e3779b9;
  let h2 = 0x85ebca6b;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ code, 0x01000193) >>> 0;
    h2 = Math.imul(h2 + code, 0x27d4eb2f) >>> 0;
  }
  const hex = (value: number) => value.toString(16).padStart(8, "0");
  return `weak-${hex(h1)}${hex(h2)}${hex((h1 ^ h2) >>> 0)}${hex((h1 + h2) >>> 0)}`;
}

export async function sha256Hex(data: string | ArrayBuffer): Promise<string> {
  const buffer =
    typeof data === "string" ? new TextEncoder().encode(data).buffer : (data as ArrayBuffer);
  if (typeof crypto !== "undefined" && crypto.subtle) {
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }
  return weakHash(typeof data === "string" ? data : new TextDecoder().decode(buffer));
}
