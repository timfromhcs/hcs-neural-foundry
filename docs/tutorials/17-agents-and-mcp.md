# Agent tool use and MCP

**Goal.** Design a narrow tool surface with approval gates.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

An agent is a model plus a harness: tool schemas, permissions, retries, memory and a stopping rule.
The harness is where safety lives, not the prompt.

The Model Context Protocol gives a typed way to expose tools and resources. The design rule used here:
read-only tools are always available; anything that spends compute, writes files or mutates weights
requires an explicit approval list, and no tool is a general shell.

## The mathematics

```
Tool contract: name, typed arguments, typed result, explicit failure mode.
Approval gate: `GATED_TOOLS` ∩ approved set, otherwise refuse before any side effect.
```

## Run it

```powershell
hcs-foundry mcp --approve foundry_start_evaluation
# without --approve, gated tools return a security_violation payload
```

## Exercise

Add a new read-only MCP tool and check that it appears in `server.list_tools()` without an approval entry.

## What you should observe

The audit log (`foundry_list_tool_calls`) records every invocation, approved or not.

## Common mistake

Adding a `run_shell` tool 'for flexibility'. It converts every prompt-injection risk into host compromise.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
