/** Curriculum and reference map. Long-form lesson prose lives in the repository (see the note). */

import { useI18n, useT } from "../i18n";
import { Alert, Card, PageHead } from "../components/ui";

interface Lesson {
  index: number;
  title: string;
  goal: string;
  minutes: number;
  path: string;
}

const LESSONS: Lesson[] = [
  { index: 1, title: "What is a neural network?", goal: "Distinguish architecture, parameters and activations.", minutes: 12, path: "docs/tutorials/01-what-is-a-neural-network.md" },
  { index: 2, title: "Tensors, parameters, layers", goal: "Read a tensor shape and predict its element count.", minutes: 15, path: "docs/tutorials/02-tensors-and-layers.md" },
  { index: 3, title: "How a parameter count is calculated", goal: "Reproduce the exact count for an MLP and a transformer.", minutes: 18, path: "docs/tutorials/03-parameter-counts.md" },
  { index: 4, title: "How a model produces an output", goal: "Trace one forward pass, including attention and RoPE.", minutes: 20, path: "docs/tutorials/04-forward-pass.md" },
  { index: 5, title: "Random initialisation and why it works", goal: "Explain fan-in scaling and reproduce a seed.", minutes: 14, path: "docs/tutorials/05-initialisation.md" },
  { index: 6, title: "Manual numerical weight editing", goal: "Apply a patch, read the diff, roll back safely.", minutes: 16, path: "docs/tutorials/06-weight-editing.md" },
  { index: 7, title: "Loss functions and gradients", goal: "Compute cross-entropy and verify a gradient numerically.", minutes: 22, path: "docs/tutorials/07-loss-and-gradients.md" },
  { index: 8, title: "Backpropagation and AdamW", goal: "Follow one optimiser step end to end.", minutes: 25, path: "docs/tutorials/08-backprop-adamw.md" },
  { index: 9, title: "Splits and overfitting", goal: "Detect overfitting instead of celebrating a low training loss.", minutes: 18, path: "docs/tutorials/09-splits-overfitting.md" },
  { index: 10, title: "Fine-tuning and LoRA", goal: "Understand low-rank adapters and their limits.", minutes: 20, path: "docs/tutorials/10-finetuning-lora.md" },
  { index: 11, title: "Evolution strategies", goal: "Run an antithetic ES step and read the fair-budget caveat.", minutes: 24, path: "docs/tutorials/11-evolution-strategies.md" },
  { index: 12, title: "Reinforcement learning and rewards", goal: "Separate DPO, PPO, GRPO and verifiable rewards.", minutes: 26, path: "docs/tutorials/12-reinforcement-learning.md" },
  { index: 13, title: "Hypernetworks and weight space", goal: "Understand what a weight generator can and cannot do.", minutes: 24, path: "docs/tutorials/13-hypernetworks.md" },
  { index: 14, title: "Diffusion and flow matching in weight space", goal: "Read the D2NWG and DeepWeightFlow claims critically.", minutes: 28, path: "docs/tutorials/14-weight-space-generation.md" },
  { index: 15, title: "Autoregressive vs diffusion language models", goal: "Compare decoding, cost and evaluation honestly.", minutes: 22, path: "docs/tutorials/15-language-model-families.md" },
  { index: 16, title: "Latent image diffusion", goal: "Identify the components of a text-to-image pipeline.", minutes: 24, path: "docs/tutorials/16-image-diffusion.md" },
  { index: 17, title: "Agent tool use and MCP", goal: "Design a narrow tool surface with approval gates.", minutes: 20, path: "docs/tutorials/17-agents-and-mcp.md" },
  { index: 18, title: "Benchmarking and reproducibility", goal: "Write a protocol someone else can rerun.", minutes: 22, path: "docs/tutorials/18-benchmarking.md" },
  { index: 19, title: "Quantisation, ONNX and the browser", goal: "Verify parity instead of assuming it.", minutes: 20, path: "docs/tutorials/19-onnx-and-browser.md" },
  { index: 20, title: "Publish a reproducible model project", goal: "Ship artifacts with provenance and honest labels.", minutes: 25, path: "docs/tutorials/20-releasing.md" },
];

export function DocsPage() {
  const t = useT();
  const { locale } = useI18n();

  return (
    <>
      <PageHead title={t("docs.title")} subtitle={t("docs.subtitle")} />

      <Alert tone="warn" title={t("docs.translationNote")}>
        {t("docs.lessonBody")}
      </Alert>

      <div style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("docs.lesson")} subtitle={t("docs.claimRule")}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>{t("docs.lesson")}</th>
                  <th>{t("docs.goal")}</th>
                  <th className="num">{t("docs.duration")}</th>
                  <th>{t("docs.reference")}</th>
                </tr>
              </thead>
              <tbody>
                {LESSONS.map((lesson) => (
                  <tr key={lesson.index}>
                    <td className="num">{lesson.index}</td>
                    <td>{lesson.title}</td>
                    <td className="tiny muted">{lesson.goal}</td>
                    <td className="num tiny">{lesson.minutes} min</td>
                    <td className="mono tiny">{lesson.path}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="tiny faint" style={{ marginTop: "var(--space-3)" }}>
            {locale === "de"
              ? "Die Lektionstexte liegen im Repository; dieser Build zeigt die Landkarte und die Ziele."
              : locale === "fr"
                ? "Les textes des leçons sont dans le dépôt ; cette version montre la carte et les objectifs."
                : "Lesson bodies live in the repository; this build shows the map and the goals."}
          </p>
        </Card>
      </div>

      <div className="split" style={{ marginTop: "var(--space-4)" }}>
        <Card title={t("docs.reference")}>
          <ul className="stack" style={{ paddingLeft: "1.1rem", margin: 0 }}>
            <li><span className="mono tiny">docs/RESEARCH.md</span> — evidence map and claims we must not make</li>
            <li><span className="mono tiny">docs/ARCHITECTURE.md</span> — modules, data flow, boundaries</li>
            <li><span className="mono tiny">docs/TESTING.md</span> — test layers and how to reproduce gates</li>
            <li><span className="mono tiny">docs/SECURITY.md</span> — threat model and sandbox rules</li>
            <li><span className="mono tiny">docs/STATUS.md</span> — generated capability matrix</li>
            <li><span className="mono tiny">DECISIONS.md</span> — architecture decision records</li>
          </ul>
        </Card>
        <Card title={t("docs.claimRule")}>
          <p className="muted">
            A capability statement needs a run: dataset hash, seed, evaluator version, hardware and the exact config.
            Without those, the honest answer is "not measured".
          </p>
          <pre className="code">{`hcs-foundry history --model my-lm
hcs-foundry eval --model my-lm --suite text-smoke --seed 1234`}</pre>
        </Card>
      </div>
    </>
  );
}
