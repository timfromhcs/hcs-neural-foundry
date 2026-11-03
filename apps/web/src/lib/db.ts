/**
 * IndexedDB persistence. No server, no sync, no telemetry.
 *
 * The schema is intentionally shallow: one object store per entity. Export/import is a plain JSON
 * bundle so a user can move a workspace between browsers and machines without a backend.
 */

import type {
  CheckpointRecord,
  EventRecord,
  HistoryBundle,
  ModelRecord,
  RunRecord,
} from "./types";

export const DB_NAME = "hcs-foundry";
export const DB_VERSION = 1;

export type StoreName = "models" | "checkpoints" | "runs" | "events" | "settings";

const STORES: { name: StoreName; keyPath: string; indexes?: { name: string; keyPath: string }[] }[] = [
  { name: "models", keyPath: "id", indexes: [{ name: "byCreatedAt", keyPath: "createdAt" }] },
  {
    name: "checkpoints",
    keyPath: "id",
    indexes: [
      { name: "byModel", keyPath: "modelId" },
      { name: "byCreatedAt", keyPath: "createdAt" },
    ],
  },
  {
    name: "runs",
    keyPath: "id",
    indexes: [
      { name: "byModel", keyPath: "modelId" },
      { name: "byStartedAt", keyPath: "startedAt" },
    ],
  },
  { name: "events", keyPath: "id", indexes: [{ name: "byAt", keyPath: "at" }] },
  { name: "settings", keyPath: "key" },
];

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

export function openDatabase(factory: IDBFactory = indexedDB): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      for (const store of STORES) {
        if (!db.objectStoreNames.contains(store.name)) {
          const created = db.createObjectStore(store.name, { keyPath: store.keyPath });
          for (const index of store.indexes ?? []) {
            created.createIndex(index.name, index.keyPath);
          }
        }
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("cannot open IndexedDB"));
    request.onblocked = () => reject(new Error("IndexedDB upgrade blocked by another tab"));
  });
}

export class Workspace {
  constructor(private readonly db: IDBDatabase) {}

  static async open(factory: IDBFactory = indexedDB): Promise<Workspace> {
    return new Workspace(await openDatabase(factory));
  }

  private async tx<T>(store: StoreName, mode: IDBTransactionMode, run: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T | undefined> {
    const transaction = this.db.transaction(store, mode);
    const objectStore = transaction.objectStore(store);
    const result = run(objectStore);
    const value = result ? await requestToPromise(result) : undefined;
    await new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("transaction failed"));
      transaction.onabort = () => reject(transaction.error ?? new Error("transaction aborted"));
    });
    return value;
  }

  // -- models ---------------------------------------------------------------------------------
  async putModel(model: ModelRecord): Promise<void> {
    await this.tx("models", "readwrite", (store) => store.put(model));
  }

  async getModel(id: string): Promise<ModelRecord | undefined> {
    return (await this.tx<ModelRecord>("models", "readonly", (store) => store.get(id))) ?? undefined;
  }

  async listModels(): Promise<ModelRecord[]> {
    const all = (await this.tx<ModelRecord[]>("models", "readonly", (store) => store.getAll())) ?? [];
    return all.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }

  // -- checkpoints ----------------------------------------------------------------------------
  async putCheckpoint(checkpoint: CheckpointRecord): Promise<void> {
    await this.tx("checkpoints", "readwrite", (store) => store.put(checkpoint));
  }

  async listCheckpoints(modelId?: string): Promise<CheckpointRecord[]> {
    const all =
      (await this.tx<CheckpointRecord[]>("checkpoints", "readonly", (store) => store.getAll())) ?? [];
    const filtered = modelId ? all.filter((item) => item.modelId === modelId) : all;
    return filtered.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }

  async latestCheckpoint(modelId: string): Promise<CheckpointRecord | undefined> {
    return (await this.listCheckpoints(modelId))[0];
  }

  // -- runs -----------------------------------------------------------------------------------
  async putRun(run: RunRecord): Promise<void> {
    await this.tx("runs", "readwrite", (store) => store.put(run));
  }

  async listRuns(modelId?: string): Promise<RunRecord[]> {
    const all = (await this.tx<RunRecord[]>("runs", "readonly", (store) => store.getAll())) ?? [];
    const filtered = modelId ? all.filter((item) => item.modelId === modelId) : all;
    return filtered.sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1));
  }

  // -- events ---------------------------------------------------------------------------------
  async logEvent(event: EventRecord): Promise<void> {
    await this.tx("events", "readwrite", (store) => store.put(event));
  }

  async listEvents(limit = 100): Promise<EventRecord[]> {
    const all = (await this.tx<EventRecord[]>("events", "readonly", (store) => store.getAll())) ?? [];
    return all.sort((a, b) => (a.at < b.at ? 1 : -1)).slice(0, limit);
  }

  // -- settings -------------------------------------------------------------------------------
  async getSetting<T>(key: string, fallback: T): Promise<T> {
    const record = await this.tx<{ key: string; value: T }>("settings", "readonly", (store) => store.get(key));
    return record ? record.value : fallback;
  }

  async setSetting<T>(key: string, value: T): Promise<void> {
    await this.tx("settings", "readwrite", (store) => store.put({ key, value }));
  }

  // -- maintenance ----------------------------------------------------------------------------
  async clearAll(): Promise<void> {
    for (const store of STORES) {
      await this.tx(store.name, "readwrite", (s) => s.clear());
    }
  }

  async exportBundle(): Promise<HistoryBundle> {
    return {
      schemaVersion: "1.0",
      exportedAt: new Date().toISOString(),
      models: (await this.tx<ModelRecord[]>("models", "readonly", (s) => s.getAll())) ?? [],
      checkpoints: (await this.tx<CheckpointRecord[]>("checkpoints", "readonly", (s) => s.getAll())) ?? [],
      runs: (await this.tx<RunRecord[]>("runs", "readonly", (s) => s.getAll())) ?? [],
      events: (await this.tx<EventRecord[]>("events", "readonly", (s) => s.getAll())) ?? [],
    };
  }

  async importBundle(bundle: HistoryBundle): Promise<void> {
    if (bundle.schemaVersion !== "1.0" || !Array.isArray(bundle.models)) {
      throw new Error("unsupported history bundle");
    }
    for (const model of bundle.models) await this.putModel(model);
    for (const checkpoint of bundle.checkpoints ?? []) await this.putCheckpoint(checkpoint);
    for (const run of bundle.runs ?? []) await this.putRun(run);
    for (const event of bundle.events ?? []) await this.logEvent(event);
  }
}

export function isHistoryBundle(value: unknown): value is HistoryBundle {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<HistoryBundle>;
  return candidate.schemaVersion === "1.0" && Array.isArray(candidate.models);
}

export async function storageEstimate(): Promise<{ used: number; quota: number }> {
  if (typeof navigator === "undefined" || !navigator.storage?.estimate) {
    return { used: 0, quota: 0 };
  }
  const estimate = await navigator.storage.estimate();
  return { used: estimate.usage ?? 0, quota: estimate.quota ?? 0 };
}

export async function persistenceGranted(): Promise<boolean> {
  if (typeof navigator === "undefined" || !navigator.storage?.persisted) return false;
  return navigator.storage.persisted();
}

export async function requestPersistence(): Promise<boolean> {
  if (typeof navigator === "undefined" || !navigator.storage?.persist) return false;
  return navigator.storage.persist();
}
