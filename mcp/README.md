# MCP server

The implementation lives in [`python/hcs_foundry/mcp_server.py`](../python/hcs_foundry/mcp_server.py).
This directory exists for client configuration examples so they stay separate from the package.

## Run it

```powershell
hcs-foundry mcp                      # read-only tools only
hcs-foundry mcp --approve foundry_start_evaluation
```

Transport: **stdio**, which is what desktop hosts use. The server binds nothing and opens no port.

## Client configuration

Claude Desktop, Cursor, VS Code and other hosts use the same shape. Replace the paths with yours:

```json
{
  "mcpServers": {
    "hcs-neural-foundry": {
      "command": "E:\\neuralfond\\.venv\\Scripts\\hcs-foundry.exe",
      "args": ["mcp", "--home", "E:\\neuralfond\\workspace"],
      "env": {}
    }
  }
}
```

Notes:

* `--home` points at a workspace; omit it to use `~/.hcs-foundry`.
* Approvals are per tool name, not global. `HCS_FOUNDRY_MCP_APPROVE` accepts a comma-separated list;
  `*` approves everything and exists only for CI.
* Every invocation is written to the `tool_invocations` table and readable via `foundry_list_tool_calls`.

## Tool surface

Read-only (13): hardware profile, capabilities, list/inspect models, list tensors, read a tensor slice,
validate a patch, solve a budget, run status, history, evaluation report, resource estimate, tool-call
log.

Approval-gated (7): start evaluation, start training, propose a weight patch, apply an approved
proposal, export ONNX, evolve, generate weights.

Deliberately absent: shell execution, arbitrary file read/write, arbitrary Python, network access.

## Example session

```text
foundry_get_hardware_profile          -> backends with declared/usable flags
foundry_solve_parameter_budget 250000  -> verified candidate architectures
foundry_inspect_model demo-char-lm     -> exact parameter breakdown
foundry_read_tensor_slice ...          -> bounded values, never a whole checkpoint
foundry_propose_weight_patch ...       -> pending proposal, policy-checked
foundry_run_weight_patch <id>          -> only with --approve, audited
```
