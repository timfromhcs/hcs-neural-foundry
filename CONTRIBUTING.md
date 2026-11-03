# Contributing

Thanks for taking this project seriously. It exists to make weight-space research **checkable**, so the
contribution rules are mostly about evidence.

## The one rule

A change that adds a capability must come with: implementation, validation, error handling, a test, and
documentation. If one of those is missing, the capability is not added — or it is added and labelled
`unsupported` in `docs/STATUS.md` and the UI.

## Setup

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[torch,export,mcp,dev]"
cd apps\web; npm install
```

## Before you open a pull request

```powershell
.\scripts\verify.ps1
```

That runs: ruff, pytest, web typecheck, vitest, production build, docs presence and an end-to-end
smoke workflow in a throwaway workspace. CI runs the same gates on Linux and Windows.

## Code conventions

* Python 3.11+ syntax, type hints on public functions, `ruff` with the project configuration.
* No `eval`, `exec`, `pickle.load`, `os.system` or `shell=True` in runtime modules — a test enforces it.
* Schemas are pydantic models with `extra="forbid"`; new fields need a migration or a schema version bump.
* Every user-visible string goes through `t(key)` and needs entries in `en`, `de` and `fr`. The parity
  test fails otherwise.
* Comments explain *why*. A comment that restates the code is noise.

## Scientific conventions

* Never report a number without: dataset hash, seed, evaluator version, hardware, and the exact config.
* Never compare methods with unequal budgets without saying so in the same paragraph.
* Never present a synthetic-corpus result as a capability result.
* Failed runs stay in the record. A run whose evaluator broke is `invalid`, not `0.0`.
* If you cannot implement a method credibly, add it to `docs/STATUS.md` as `unsupported` and explain
  what would be required. That is a contribution, not a gap.

## Commit messages

Conventional-ish, scoped by area:

```
feat(weights): add structured row sparsity
fix(api): reject non-loopback token by default
docs(research): correct the D2NWG scope statement
test(training): cover cancel between steps
```

## Reporting a bug

Include the output of `hcs-foundry doctor --json`, the exact command, and the run id if the failure
happened inside a run. Do not paste tokens; the API token is not needed to reproduce anything.

## License

By contributing you agree that your contribution is licensed under the MIT licence of this repository.
