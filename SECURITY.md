# Security policy

The full threat model, controls and residual risks are in [`docs/SECURITY.md`](docs/SECURITY.md).
This file is the short version.

## Reporting a vulnerability

Please report privately — a security advisory on the repository, or an email to the maintainers listed
in `pyproject.toml`. Do not open a public issue for an unpatched vulnerability.

Include:

* affected version (`hcs-foundry doctor --json` prints it),
* a minimal reproduction,
* what an attacker gains (data exfiltration, code execution, integrity, denial of service),
* whether the local API or MCP server is involved.

## Supported versions

`0.1.x` receives fixes. This is an alpha; breaking changes are documented in `CHANGELOG.md`.

## What is in scope

* Local HTTP API authentication, CORS and rate limiting.
* Patch validation and checkpoint I/O (malformed tensors, size caps, path handling).
* The agent policy layer and the isolated verifier boundary.
* MCP tool exposure (no shell, no arbitrary filesystem access).
* Browser storage handling and the optional local-engine client.

## What is out of scope

* Training quality, model capability, benchmark numbers.
* Vulnerabilities in PyTorch, ONNX Runtime or the browser itself (report upstream; we still want to know).
* Anything requiring an attacker who already has code execution on the host.
* Denial of service against yourself with deliberately impossible budgets.

## Defaults that protect you

| Default | Value | Change it if |
|---|---|---|
| API bind address | `127.0.0.1` | you understand the exposure and put auth/TLS in front |
| API token | random, stored `0600` in the workspace | rotate by deleting the file |
| CORS | localhost dev origins only | you host the UI elsewhere |
| MCP gated tools | none approved | you reviewed the specific tool call |
| Checkpoint format | safetensors only | never — pickle import is unsupported by design |
| Telemetry | none | — |
