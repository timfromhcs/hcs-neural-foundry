# Examples

Small, runnable demonstrations. Every example uses a throwaway workspace so it never touches your
default `~/.hcs-foundry`.

| Example | What it shows | Command |
|---|---|---|
| [`tiny-mlp`](tiny-mlp/) | Create, train, evaluate and export a 676-parameter MLP | `python examples/tiny-mlp/run.py` |
| [`es-vs-backprop`](es-vs-backprop/) | Evolution Strategies against gradient training on the same example budget | `python examples/es-vs-backprop/run.py` |
| [`weight-patch`](weight-patch/) | Apply a structured patch, measure its effect, roll back | `python examples/weight-patch/run.py` |

All three scripts print a table and leave a workspace behind (path printed) so you can inspect the
database and checkpoints with the CLI:

```powershell
hcs-foundry --home <printed-path> history
```
