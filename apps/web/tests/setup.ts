import "fake-indexeddb/auto";

// jsdom does not implement matchMedia; several components read it indirectly via CSS only, but
// keeping a stub avoids surprises if a component starts querying it.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList;
}

if (!window.crypto?.subtle) {
  // Tests exercise the deterministic fallback path on purpose.
  Object.defineProperty(window, "crypto", {
    value: { getRandomValues: (array: Uint8Array) => array.fill(7) },
    configurable: true,
  });
}
