# Reinforcement learning and reward signals

**Goal.** Separate DPO, PPO, GRPO and verifiable rewards.

**Time.** about 20 minutes. **Prerequisites.** earlier lessons in this folder.

## Intuition

RL post-training optimises behaviour after a model has basic competence.

* **DPO** — preference optimisation on pairs; no online rollouts, no reward model at train time.
* **PPO** — policy gradient with a clipped update and a KL constraint to a reference policy.
* **GRPO** — sample several completions per prompt and use relative reward inside the group; no value
  network.
* **RLVR** — rewards from automatically checkable outcomes (maths answers, tests passing).

The failure mode that matters is reward hacking: a strong optimiser will find the cheapest way to
increase the score, including editing the test.

## The mathematics

```
GRPO advantage for completion `i`: `A_i = (r_i - mean(r)) / std(r)`.
KL penalty keeps `π` close to `π_ref`; without it, the model collapses onto the reward's quirks.
```

## Run it

```powershell
# The evaluator fixtures and reward interface exist; no RL trainer is shipped.
hcs-foundry eval --model lesson-3 --suite weight-sanity
hcs-foundry proposal create --model lesson-3 --agent agent:demo --goal "shrink norms" --tensor blocks.0.attn.q_proj.weight --op scale --params '{"factor":0.95}'
```

## Exercise

Write the reward function you would use for a verifiable task, then write the exploit a policy would find.

## What you should observe

Proposals are immutable and the verifier runs in a separate process — the structure that makes reward tampering hard.

## Common mistake

Reporting reward improvement as capability improvement. Reward is a proxy; held-out task success is the measurement.

---

Next: see the following lesson. All commands use a throwaway workspace when you set
`$env:HCS_FOUNDRY_HOME` first; nothing here modifies your default workspace by accident.
