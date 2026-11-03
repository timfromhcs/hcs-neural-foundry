# Curriculum

Twenty lessons, from "what is a parameter" to shipping a reproducible model project. Each lesson
follows the same contract: goal, intuition, the mathematics, a runnable command, an exercise, what you
should observe, and the mistake that lesson exists to prevent.

| # | Lesson | Goal |
|---|---|---|
| 1 | [What is a neural network?](01-what-is-a-neural-network.md) | Separate architecture, parameters and activations |
| 2 | [Tensors, parameters, layers](02-tensors-and-layers.md) | Read shapes and predict element counts |
| 3 | [How a parameter count is calculated](03-parameter-counts.md) | Reproduce exact counts for MLP and transformer |
| 4 | [How a model produces an output](04-forward-pass.md) | Trace attention, RoPE and masked prediction |
| 5 | [Random initialisation](05-initialisation.md) | Explain fan-in scaling, reproduce a seed |
| 6 | [Manual weight editing](06-weight-editing.md) | Patch, measure and roll back |
| 7 | [Loss and gradients](07-loss-and-gradients.md) | Cross-entropy and numeric gradient checks |
| 8 | [Backpropagation and AdamW](08-backprop-adamw.md) | Follow one optimiser step |
| 9 | [Splits and overfitting](09-splits-overfitting.md) | Detect overfitting honestly |
| 10 | [Fine-tuning and LoRA](10-finetuning-lora.md) | Low-rank adapters and their limits |
| 11 | [Evolution Strategies](11-evolution-strategies.md) | Antithetic ES and fair budgets |
| 12 | [Reinforcement learning](12-reinforcement-learning.md) | DPO/PPO/GRPO/RLVR and reward hacking |
| 13 | [Hypernetworks](13-hypernetworks.md) | What a weight generator can and cannot do |
| 14 | [Weight-space diffusion and flow matching](14-weight-space-generation.md) | Read the claims critically |
| 15 | [Language model families](15-language-model-families.md) | Autoregressive vs diffusion |
| 16 | [Latent image diffusion](16-image-diffusion.md) | Pipeline components and versioning |
| 17 | [Agents and MCP](17-agents-and-mcp.md) | Narrow tools, approval gates |
| 18 | [Benchmarking](18-benchmarking.md) | Protocols someone else can rerun |
| 19 | [ONNX and the browser](19-onnx-and-browser.md) | Verify parity, name the provider |
| 20 | [Releasing](20-releasing.md) | Artifacts, model cards, honest labels |

Lessons 12, 14, 15, 16 describe capabilities this build declares **unsupported**. They teach the
concepts and state plainly what is missing — a lesson about a method is not a claim that the method
is implemented here.

**Made with ❤️ by hcsmedia and timfromhcs**
