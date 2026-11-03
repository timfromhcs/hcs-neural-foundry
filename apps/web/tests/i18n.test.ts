import { describe, expect, it } from "vitest";

import { dictionaries, LOCALES } from "../src/locales";
import { en } from "../src/locales/en";
import { interpolate } from "../src/i18n";

const keys = Object.keys(en).sort();

describe("locale parity", () => {
  it("ships exactly the source keys in every locale", () => {
    for (const locale of LOCALES) {
      expect(Object.keys(dictionaries[locale]).sort(), `locale ${locale}`).toEqual(keys);
    }
  });

  it("has no empty or placeholder-only translations", () => {
    for (const locale of LOCALES) {
      for (const key of keys) {
        const value = (dictionaries[locale] as Record<string, string>)[key] ?? "";
        expect(value, `${locale}:${key}`).toBeTruthy();
        expect(value.trim().length, `${locale}:${key}`).toBeGreaterThan(0);
        expect(value, `${locale}:${key}`).not.toBe("TODO");
      }
    }
  });

  it("keeps technical identifiers untranslated where it matters", () => {
    for (const locale of LOCALES) {
      const dictionary = dictionaries[locale] as Record<string, string>;
      expect(dictionary["app.name"]).toBe("HCS Neural Foundry");
      expect(dictionary["api.title"]).toContain("MCP");
    }
  });

  it("does not leave HTML in translation values", () => {
    for (const locale of LOCALES) {
      for (const key of keys) {
        const value = (dictionaries[locale] as Record<string, string>)[key];
        expect(value, `${locale}:${key}`).not.toMatch(/<[a-z][^>]*>/i);
      }
    }
  });
});

describe("interpolation", () => {
  it("replaces named placeholders", () => {
    expect(interpolate("hello {name}", { name: "world" })).toBe("hello world");
    expect(interpolate("n={n}", { n: 3 })).toBe("n=3");
  });

  it("leaves unknown placeholders visible instead of printing undefined", () => {
    expect(interpolate("hello {missing}")).toBe("hello {missing}");
  });

  it("returns the template unchanged without values", () => {
    expect(interpolate("plain")).toBe("plain");
  });
});
