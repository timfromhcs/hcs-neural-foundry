/**
 * Minimal, typed i18n. No dependency, no runtime key lookup failures: `t` only accepts keys that exist
 * in the English source dictionary, and a test asserts that de/fr cover exactly those keys.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { dictionaries, LOCALES, type Dictionary, type Locale, type TranslationKey } from "./locales";

const STORAGE_KEY = "hcs-foundry.locale";

export type Translate = (key: TranslationKey, values?: Record<string, string | number>) => string;

interface I18nValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: Translate;
}

const I18nContext = createContext<I18nValue | null>(null);

function isLocale(value: string | null | undefined): value is Locale {
  return value === "en" || value === "de" || value === "fr";
}

function detectLocale(): Locale {
  if (typeof window === "undefined") return "en";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (isLocale(stored)) return stored;
  const languages = navigator.languages ?? [navigator.language];
  for (const language of languages) {
    const primary = language.slice(0, 2).toLowerCase();
    if (isLocale(primary)) return primary;
  }
  return "en";
}

export function interpolate(template: string, values?: Record<string, string | number>): string {
  if (!values) return template;
  return template.replace(/\{(\w+)\}/g, (match, key: string) => {
    const value = values[key];
    return value === undefined ? match : String(value);
  });
}

export function I18nProvider({ children, initialLocale }: { children: ReactNode; initialLocale?: Locale }) {
  const [locale, setLocaleState] = useState<Locale>(() => initialLocale ?? detectLocale());

  useEffect(() => {
    document.documentElement.lang = locale;
    try {
      window.localStorage.setItem(STORAGE_KEY, locale);
    } catch {
      // Private mode with blocked storage: the UI still works, it just forgets the choice.
    }
  }, [locale]);

  const setLocale = useCallback((next: Locale) => {
    if (LOCALES.includes(next)) setLocaleState(next);
  }, []);

  const value = useMemo<I18nValue>(() => {
    const dictionary: Dictionary = dictionaries[locale];
    const t: Translate = (key, values) => interpolate(dictionary[key] ?? dictionaries.en[key] ?? key, values);
    return { locale, setLocale, t };
  }, [locale, setLocale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const context = useContext(I18nContext);
  if (!context) throw new Error("useI18n must be used inside <I18nProvider>");
  return context;
}

export function useT(): Translate {
  return useI18n().t;
}
