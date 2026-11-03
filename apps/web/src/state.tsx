/**
 * Application state: one IndexedDB workspace, one demo manifest, one preferences object.
 *
 * Preferences are applied as attributes on <html> so CSS tokens react immediately; they are persisted
 * in localStorage because they are UI state, not experiment data.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { Workspace } from "./lib/db";
import { loadDemoManifest } from "./lib/demo";
import type { DemoManifest } from "./lib/types";

export interface Preferences {
  theme: "dark" | "light";
  accent: "blue" | "violet" | "emerald" | "amber" | "rose";
  density: "comfortable" | "compact";
  contrast: boolean;
  provider: "auto" | "wasm" | "webgpu";
}

export const DEFAULT_PREFERENCES: Preferences = {
  theme: "dark",
  accent: "blue",
  density: "comfortable",
  contrast: false,
  provider: "auto",
};

const PREF_KEY = "hcs-foundry.preferences";

interface AppContextValue {
  db: Workspace | null;
  demo: DemoManifest | null;
  demoError: boolean;
  ready: boolean;
  version: number;
  preferences: Preferences;
  setPreferences: (next: Partial<Preferences>) => void;
  refresh: () => void;
}

const AppContext = createContext<AppContextValue | null>(null);

function readPreferences(): Preferences {
  if (typeof window === "undefined") return DEFAULT_PREFERENCES;
  try {
    const raw = window.localStorage.getItem(PREF_KEY);
    if (!raw) return DEFAULT_PREFERENCES;
    const parsed = JSON.parse(raw) as Partial<Preferences>;
    return { ...DEFAULT_PREFERENCES, ...parsed };
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

function applyPreferences(preferences: Preferences) {
  const root = document.documentElement;
  root.dataset.theme = preferences.theme;
  root.dataset.accent = preferences.accent;
  root.dataset.density = preferences.density;
  root.dataset.contrast = preferences.contrast ? "high" : "normal";
  root.style.colorScheme = preferences.theme;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [db, setDb] = useState<Workspace | null>(null);
  const [demo, setDemo] = useState<DemoManifest | null>(null);
  const [demoError, setDemoError] = useState(false);
  const [ready, setReady] = useState(false);
  const [version, setVersion] = useState(0);
  const [preferences, setPreferencesState] = useState<Preferences>(() => readPreferences());

  const refresh = useCallback(() => setVersion((value) => value + 1), []);

  const setPreferences = useCallback((next: Partial<Preferences>) => {
    setPreferencesState((current) => {
      const merged = { ...current, ...next };
      applyPreferences(merged);
      try {
        window.localStorage.setItem(PREF_KEY, JSON.stringify(merged));
      } catch {
        // Storage may be blocked; preferences still apply for this session.
      }
      return merged;
    });
  }, []);

  useEffect(() => {
    applyPreferences(preferences);
  }, [preferences]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const workspace = await Workspace.open();
        if (!cancelled) setDb(workspace);
      } catch {
        if (!cancelled) setDb(null);
      }
      const manifest = await loadDemoManifest();
      if (!cancelled) {
        setDemo(manifest);
        setDemoError(manifest === null);
        setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo<AppContextValue>(
    () => ({
      db,
      demo,
      demoError,
      ready,
      version,
      preferences,
      setPreferences,
      refresh,
    }),
    [db, demo, demoError, ready, version, preferences, setPreferences, refresh],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppContextValue {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside <AppProvider>");
  return context;
}

/** Convenience: the workspace, or null while IndexedDB is still opening. */
export function useWorkspace(): Workspace | null {
  return useApp().db;
}
