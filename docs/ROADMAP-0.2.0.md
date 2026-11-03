# Roadmap 0.2.0 — from `0.1.0` alpha to first published stable

> Status: **execution contract for this milestone.** It is not a wish list. Every task below has a
> goal, a test that can fail, an acceptance gate and a rollback. A task leaves this file only when its
> gate passes on this machine *and* in CI.
>
> Owner: autonomous build agent. Human approval is required for the three external actions flagged
> **APPROVAL** (repository creation, deployment, tag/release).
>
> Written: 2026-11-03. Target: `0.2.0` released and deployed.

---

## 1. Why 0.2.0 exists at all

`0.1.0` is a complete vertical slice: 118 Python tests, 40 web tests, 12 weight operations, a real
trainer, a real GRPO loop, a real flow-matching generator, and honest negative results published in
`docs/PERFORMANCE.md`. What it is **not** is verified against the software it claims to speak to.
Two concrete drifts were found while auditing this repository, and each of them invalidates a headline
claim:

| # | Drift | Evidence found in the repo | Consequence |
|---|---|---|---|
| D1 | The MCP server is written against SDK **v1** (`from mcp.server.fastmcp import FastMCP`, `mcp>=1.2,<2`). | `python/hcs_foundry/mcp_server.py:63`, `pyproject.toml` extras | SDK v1 is in maintenance mode. `AGENTS.md` §4.3 targets the 2026-07-28 revision. The server cannot be described as "spec 2026-07-28". |
| D2 | The browser runtime imports bare `onnxruntime-web`, i.e. the **JSEP** WebGPU backend. | `apps/web/src/lib/onnx.ts:31,42` (`type OrtModule = typeof import("onnxruntime-web")`) | ONNX Runtime **deprecated WebGL and JSEP in v1.29.0** (merged 2026-08-04, PR #29716). The "WebGPU works" claim runs on a removed path. |

Everything else in 0.2.0 exists to make those two fixes safely publishable: a release gate that
cannot be skipped, a version that is consistent in five places, and a deploy whose numbers are
measured rather than asserted.

## 2. Research findings that drive the work (open questions, answered)

All links verified on 2026-11-03. These are the answers that convert uncertain items into tasks or
explicit non-goals.

### Q1 — Which MCP SDK line is current, and can it serve both protocol revisions?

**Answer:** `mcp` **2.3.0** is the current stable. v2.0.0 shipped 2026-07-28 and speaks the
**2026-07-28** revision; `pip install mcp` now installs 2.x. v1.x lives on the `v1.x` branch in
maintenance mode. v2 serves *both* protocol eras from one server with no flag.

API deltas that actually matter to this codebase:

| v1 | v2 |
|---|---|
| `from mcp.server.fastmcp import FastMCP` | `from mcp.server import MCPServer` (the `fastmcp` module is **removed**) |
| `FastMCP(name, instructions=...)` | `MCPServer(name, instructions=...)` (same kwargs) |
| `@server.tool(name=...)`, `@server.resource(uri, mime_type=...)`, `@server.prompt(name)` | unchanged |
| `list_tools()` / `call_tool()` → coroutine | still coroutines |
| `call_tool` → tuple / `Tool` | `CallToolResult` with `.content[0].text` and `is_error` |
| `server.run(transport="stdio")` | unchanged |

Sources: <https://py.sdk.modelcontextprotocol.io/whats-new/>,
<https://py.sdk.modelcontextprotocol.io/migration/>,
<https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0>.

### Q2 — Is the browser WebGPU path still the supported one?

**Answer:** the *capability* is supported, the *implementation we use* is not. ONNX Runtime announced
the deprecation of WebGL and JSEP; the native WebGPU EP (`import "onnxruntime-web/webgpu"`) is the
recommended path. Both builds still ship in `onnxruntime-web@1.31.1`, and both are present in
`node_modules`, so the migration is an import + probe change, not a rewrite. WASM remains the default
and the tested fallback.

Sources: <https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html>,
<https://github.com/microsoft/onnxruntime/pull/29716>,
<https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md>.

### Q3 — Does the browser claim survive the migration?

**Unknown until measured.** The `0.1.0` figure (WebGPU, 18.2 ms) was taken on the JSEP backend. It is
therefore *re-opened*, not carried forward. T05 must produce a new number or downgrade the claim to
"WASM only, verified".

### Q4 — Is the shipped model inside the WebGPU operator set?

**Partially answerable offline.** The demo graph uses `Gather`, `MatMul`, `Add`, `LayerNormalization`,
`Softmax`, `Reshape`, `Transpose`-family ops — all present in the WebGPU operator table. That is a
*necessary* condition. The only sufficient condition is a real inference on the native EP, which is
what T05 measures.

### Q5 — Package names

`hcs-foundry` on PyPI and `@hcs-foundry/web` on npm are **not** checked or claimed in 0.2.0.
Publication is out of scope (decision Q6). `docs/STATUS.md` keeps them `unsupported` until they are
actually resolvable from a clean environment.

### Q6 — Publish scope (human decision, recorded)

GitHub + Vercel only. No PyPI, no npm. `.github/workflows/release.yml` therefore builds and attaches
artifacts instead of publishing them, and `pyproject.toml` keeps `Development Status :: 4 - Beta`
pending a real registry publication.

---

## 3. Definition of "stable" for 0.2.0

0.2.0 is done when **all** of these are true at the same commit:

1. `mcp` resolves to `>=2,<3` and `hcs-founder mcp serve` starts under SDK 2.3.0 with the approval
   gate intact.
2. The browser runtime imports the native WebGPU EP, reports a provider only after a real inference,
   and the WebGPU/WASM claim in `docs/STATUS.md` matches a measurement taken **after** the change.
3. Version `0.2.0` appears identically in five places and a script fails if they diverge.
4. `scripts/verify.ps1` is green end to end, including the smoke workflow.
5. `scripts/verify_release.py` produces a machine-readable report with zero blocking findings.
6. CI is green on `main` on both Linux and Windows.
7. The static site is deployed and returns HTTP 200 with no request to any origin other than itself.
8. The GitHub release exists with the wheel, the SBOM and the checksums attached.
9. `main` is the default branch and contains the released commit; `master` is gone.
10. `README.md` states the real state — including what is still unsupported.

Anything not on this list is explicitly not a 0.2.0 requirement.

## 4. Non-goals (deliberately not attempted)

| Item | Why not now |
|---|---|
| PyPI / npm publication | No trusted-publisher credentials verified; a half-published package is worse than an unpublished one. |
| Tauri desktop shell / single-file launcher | Unrelated to the two drifts; needs a Rust toolchain pass and clean-VM matrices. |
| Corpus-scale weight diffusion (D2NWG-class) | Needs thousands of aligned checkpoints. A toy with the paper's name on it is a false claim. |
| DPO / PPO / RLHF at token level | Needs rollout infrastructure and a reward model. GRPO-on-bandit stays the honest demo. |
| Diffusion LM / latent image diffusion | Separate model families; each needs a trainer, sampler and metrics. |
| 20-lesson curriculum translated to de/fr | UI is translated; long-form lessons are English-only and labelled as such. |
| Visual regression snapshots | axe-core is in place; pixel diffing needs a baseline store. |

---

## 5. Workstreams

Each task follows the same **loop protocol** (§6). Budgets are soft: exceeding one is a signal to
escalate, not to grind.

### T01 — Freeze the baseline and record the recon

- **Goal:** a commit that a later gate can compare against, plus a written recon of what the last
  session left open.
- **Files:** `docs/ROADMAP-0.2.0.md` (this file), `plan.md`.
- **Tests:** `pytest python/tests -q` must reproduce 118 passed before any source change.
- **Acceptance:** baseline test counts recorded in `plan.md` with a date.
- **Rollback:** revert the doc commit; no source touched.

### T02 — MCP SDK v2 migration

- **Goal:** the server runs on `mcp>=2,<3` with unchanged external behaviour.
- **Files:** `pyproject.toml`, `python/hcs_foundry/mcp_server.py`,
  `python/tests/test_interfaces.py`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`, `CHANGELOG.md`.
- **Changes:**
  - extra becomes `mcp = ["mcp>=2.3,<3"]`;
  - `FastMCP` → `MCPServer`, import path moved;
  - the missing-extra error message names the correct install line;
  - tests read `CallToolResult.content[0].text` instead of assuming a tuple, and assert
    `is_error` is falsy for a successful call;
  - a new test asserts the SDK major version is 2, so a silent downgrade fails the suite.
- **Risks:** (a) tool signature introspection differs; (b) `is_error` naming; (c) a v2-only async
  change. Mitigation: probe the installed package before editing (done: `list_tools`/`call_tool`
  remain coroutines).
- **Tests:** the three existing MCP tests + the new version assertion + `python -m hcs_foundry mcp
  serve` startup smoke (start, read one line of stderr, terminate).
- **Acceptance:** `pytest python/tests -q` green; `hcs-foundry mcp --help` exit 0; a stdio handshake
  succeeds under 2.3.0.
- **Rollback:** revert the commit; the pin `mcp>=1.2,<2` restores the old behaviour exactly.

### T03 — Native WebGPU EP in the browser runtime

- **Goal:** the browser uses the supported ONNX Runtime Web path and never reports an unverified
  provider.
- **Files:** `apps/web/src/lib/onnx.ts`, `apps/web/tests/*.test.ts(x)`, `docs/STATUS.md`,
  `docs/PERFORMANCE.md`.
- **Changes:**
  - import the WebGPU build (`onnxruntime-web/webgpu`), keeping the same WASM fallback inside it;
  - extend the probe to record `navigator.gpu` presence *and* the resolved adapter info where the
    browser exposes it, without gating on it;
  - keep the invariant "reported ⇔ a real inference ran on it";
  - a unit test that stubs a failing WebGPU session and asserts the reported provider is `wasm` and
    `fallbackReason` is populated.
- **Risks:** (a) the `webgpu` bundle needs its own wasm loading strategy; (b) bundle size; (c) a
  browser without WebGPU must still work. Mitigation: the fallback test and a manual WASM run.
- **Tests:** `npm run typecheck`, `npm test`, `npm run build`, then a real browser run (T05).
- **Acceptance:** build succeeds; the fallback test passes; no request to a CDN.
- **Rollback:** revert the import; the JSEP path still works on 1.31.1, so this is a safe revert.

### T04 — Version and documentation consistency

- **Goal:** `0.2.0` stated once and enforced.
- **Files:** `pyproject.toml`, `python/hcs_foundry/__init__.py`, `apps/web/package.json`,
  `README.md`, `CHANGELOG.md`, `docs/STATUS.md`, `docs/acceptance.json`, `plan.md`.
- **Changes:** bump, add the `[0.2.0]` changelog section, refresh the capability matrix with the v2 and
  WebGPU facts, re-date `acceptance.json`, and mark the two drifts closed in `plan.md`.
- **Tests:** new `scripts/verify_release.py` check `version_consistency`.
- **Acceptance:** the script fails if any of the five locations disagrees — prove it by temporarily
  breaking one and observing the failure.
- **Rollback:** revert.

### T05 — Measured browser verification

- **Goal:** replace the inherited 18.2 ms number with a measurement taken on the new code path, or
  state WASM-only.
- **Procedure:** build, serve `dist/`, open in Chromium, run the demo inference, record provider,
  milliseconds, and the exact browser version.
- **Tests:** manual + the recorded numbers in `docs/PERFORMANCE.md` with the date and browser build.
- **Acceptance:** the claim in `STATUS.md` cites the new measurement; if WebGPU fails, the entry is
  downgraded to WASM-only and the reason is written down.
- **Rollback:** none needed; a documentation correction is always allowed to go backwards.

### T06 — Release gate hardening

- **Goal:** one command that cannot pass while something is wrong.
- **Files:** `scripts/verify_release.py`, `scripts/release.ps1` (new), `scripts/verify.ps1`,
  `.github/workflows/ci.yml`, `.github/workflows/release.yml` (new).
- **`verify_release.py` checks:** version consistency; brand credit present in README and app;
  `acceptance.json` parses and every area has evidence or an explicit `unsupported`; no tracked
  secret patterns; tutorial count ≥ 20; `dist/index.html` exists and references no external origin;
  `CHANGELOG.md` has an entry for the current version.
- **`release.ps1` order:** ruff → pytest → typecheck → vitest → build → verify_release → encoding
  scan → dependency inventory. Fail fast, print a table.
- **Acceptance:** deliberately break each check once and confirm it fails. A gate that has never
  failed is not a gate.
- **Rollback:** revert.

### T07 — Local full-gate loop

- **Goal:** green on this machine, with the numbers recorded.
- **Tests:** `scripts/verify.ps1` (web + smoke on), `scripts/release.ps1`.
- **Acceptance:** zero failures; durations recorded in `plan.md`.
- **Loop:** §6, max 3 attempts per defect, then escalate with a diagnosis.

### T08 — Deployment (APPROVAL: human granted at session start)

- **Goal:** the static site lives at a stable production URL.
- **Steps:** `vercel link` → `vercel deploy` (preview) → verify → `vercel deploy --prod` → verify.
- **Acceptance:** HTTP 200; `index.html` loads; the model file and ORT wasm load from the same
  origin; `Content-Type: application/wasm` is served correctly; no console errors.
- **Rollback:** `vercel rollback` to the previous deployment, or remove the project.

### T09 — Repository, tag, release (APPROVAL: human granted at session start)

- **Goal:** public `timfromhcs/hcs-neural-foundry` with a `v0.2.0` release.
- **Steps:** `gh repo create --public` → push `main` → push tag `v0.2.0` → build wheel + SBOM →
  `gh release create` with artifacts and generated notes → verify the release page.
- **Acceptance:** the clone URL clones; the wheel installs into a clean venv and `doctor` runs.
- **Rollback:** delete the release, delete the tag, or make the repository private. Destructive
  steps require a fresh human confirmation.

### T10 — README as the front door

- **Goal:** a stranger can decide in 30 seconds whether this project is real.
- **Content:** one-line pitch; live URL; maturity table with the honest labels; install; the four
  commands that prove the pipeline; the "what this is not" section; the research stance; links.
- **Acceptance:** every command in the README was executed in this session; no dead links; the live
  URL is the deployed one.

### T11 — Merge to `main`

- **Goal:** one branch, one history, no divergence.
- **Steps:** fast-forward or merge `master` → `main`, set `main` as default, delete `master`, verify
  CI triggers.
- **Acceptance:** `git log --oneline` on `main` contains every commit; CI green.
- **Rollback:** recreate `master` at the pre-merge commit; nothing is force-pushed.

---

## 6. The loop protocol (applies to every task)

```
OBSERVE    read the actual state: git status, test output, logs. Never assume.
REPRODUCE  write the smallest failing check *before* the fix. If it cannot fail,
           the change is not a fix, it is a preference.
DIAGNOSE   name the cause: code defect / environment / flaky / wrong expectation /
           unsupported hardware. These have different correct responses.
PATCH      smallest diff that addresses the diagnosed cause. No drive-by refactors.
VERIFY     narrow test → related tests → full gate. A green narrow test is not a green gate.
RECORD     update plan.md, CHANGELOG.md, and the measured numbers. Update this file's
           task state so the next session inherits the truth.
```

**Stop rule:** at most **3 materially different** attempts per defect. After the third failure,
write the diagnosis, the two hypotheses already excluded, and the exact decision required. Do not
repeat an identical action and call it persistence.

**Prohibited shortcuts** (these would make the gate meaningless and are treated as failures, not
wins): deleting a test, loosening a threshold, `pytest.skip` without a tracked reason, mocking the
thing under test, editing `docs/PERFORMANCE.md` numbers without a new run, or reporting only the
best seed.

## 7. Gate matrix

| Gate | Command | Blocks |
|---|---|---|
| G1 lint | `.venv\Scripts\python.exe -m ruff check python` | any commit |
| G2 python tests | `pytest python/tests -q` | any commit |
| G3 types | `npm run typecheck` (in `apps/web`) | any commit |
| G4 web tests | `npm test` (in `apps/web`) | any commit |
| G5 web build | `npm run build` (in `apps/web`) | any commit |
| G6 smoke | `verify.ps1` end-to-end section | any commit |
| G7 release audit | `python scripts/verify_release.py --strict` | tag |
| G8 browser measurement | manual, recorded | WebGPU claim |
| G9 live site | HTTP 200 + local-only asset requests | deploy |
| G10 clean install | wheel into fresh venv, `doctor` | release |

## 8. Risk register for this milestone

| # | Risk | Trigger | Response |
|---|---|---|---|
| R10 | MCP v2 tool-schema drift breaks a client's expectations | an external host rejects the server | keep arguments identical; if a signature must change, it is a documented breaking change in CHANGELOG |
| R11 | Native WebGPU EP fails on this GPU/driver | T05 measurement fails | ship WASM-only, downgrade the claim, record the failure. Never claim WebGPU. |
| R12 | `onnxruntime-web/webgpu` inflates the bundle beyond reason | build output | measure; if it doubles, load the EP lazily and keep WASM as the eager path |
| R13 | Vercel build differs from local build | deploy fails | pin Node in `package.json` `engines`; use `npm ci` |
| R14 | Release artifacts disagree with the tag | G7 | G7 is mandatory and blocks the tag |
| R15 | A public push exposes a secret | `git grep` scan in G7 | the release gate scans tracked files before the push; rotate if hit |

## 9. Sequence and budget

| Step | Tasks | Est. | Hard stop |
|---|---|---|---|
| 1 | T01, T02, T03 | 60–90 min | MCP server starts, browser builds |
| 2 | T04, T06 | 40–60 min | gate fails when deliberately broken |
| 3 | T05, T07 | 40–60 min | full local gate green |
| 4 | T08, T09 | 30–45 min | live URL + release page |
| 5 | T10, T11 | 20–30 min | `main` is default, CI green |

## 10. What changing my mind looks like

If a task turns out to be wrong — for example if native WebGPU is genuinely broken for this model on
this hardware — the correct outcome is a **narrower but true** release, not a green lie. Downgrade the
label, write the limitation into `docs/STATUS.md` and `README.md`, and keep the pipeline honest. That
is the difference between shipping software and shipping marketing.

---

**Made with ❤️ by hcsmedia and timfromhcs**
