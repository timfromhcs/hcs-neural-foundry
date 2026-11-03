import { en } from "./en";
import { de } from "./de";
import { fr } from "./fr";
import type { Dictionary } from "./en";

export type Locale = "en" | "de" | "fr";

export const LOCALES: Locale[] = ["en", "de", "fr"];

export const dictionaries: Record<Locale, Dictionary> = { en, de, fr };

export type { Dictionary };
export type { TranslationKey } from "./en";

export const LOCALE_LABEL: Record<Locale, string> = {
  en: "English",
  de: "Deutsch",
  fr: "Français",
};
