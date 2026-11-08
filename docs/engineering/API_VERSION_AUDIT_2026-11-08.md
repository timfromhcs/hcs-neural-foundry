# API version consistency audit — 2026-11-08

Starting source: main `4635ac2018a35d3f0628154038fe29eb7d906911`. Feature branch: `codex/api-version-consistency`. Package/runtime version is 0.2.0; health and FastAPI OpenAPI metadata previously hard-coded 0.1.0.

## Change and regression proof

The health and OpenAPI interface tests were strengthened to compare their reported version with `hcs_foundry.__version__`. Both assertions failed against the original source (0.1.0 versus 0.2.0), then passed after replacing the two literals with the package version. No API payload fields, authentication behavior, package version or dependency bounds changed.

The workspace field remains because the existing browser API client explicitly consumes the current health response type and the documented contract exempts health from bearer authentication. Its unauthenticated path visibility is an existing security consideration, not resolved by this version-only fix. Browser footer version consistency is separate follow-up work; no frontend source changed here.

Current Ruff 0.17.4 also reported pre-existing SIM102 in the SET_TRAINABLE validator. A separate commit combines the equivalent condition; validation behavior remains covered by the existing full suite. No lint exclusion or dependency downgrade was added.

## Executed gates

Isolated Python 3.12.13 environment, installed from this repository only with `.[torch,export,mcp,dev]`. It is not shared with the distinct HCS Foundry distribution bearing the same package name. Resolved test dependencies include FastAPI 0.162.2, PyTorch 2.15.0, MCP 2.3.0 and Ruff 0.17.4.

- Targeted health/OpenAPI regression: 2 failed before correction; 2 passed after.
- `python -m ruff check python`: passed.
- `python -m pytest python/tests -q -rs --junitxml=artifacts/api-version-pytest.xml`: 121 tests, 0 failures/errors/skips; XML inspected.
- `npm ci --prefix apps/web`: completed, audit 0 vulnerabilities; npm retained a protobufjs install-script notice. No allowScripts configuration changed.
- `scripts/release.ps1`: exit 0; Python suite, web typecheck, 48 web tests, production build, strict release audit, encoding scan and dependency inventory passed. It publishes nothing.
- `verify.verify` was called with the repository path and full test mode. It detected Python/venv but reported `hasPytest:false`, `no runnable checks detected`, `ok:false` for the nested `python/tests` layout. That tool result is unavailable verification, not a pass; direct repository gates above are the actual execution evidence.

No cloud model invocation, model download, UI redesign, production deploy, release tag or main merge was performed.
