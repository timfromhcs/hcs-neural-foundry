/// <reference lib="webworker" />

/**
 * Training worker: keeps the main thread free while a real SGD loop runs.
 *
 * Protocol (all messages are plain objects so they survive structured cloning):
 *   in : { type: "start", config, steps, batchSize, lr, seed }
 *        { type: "stop" }
 *   out: { type: "progress", step, loss, elapsedMs }
 *        { type: "done", result, weights, elapsedMs, stopped }
 *        { type: "error", message }
 */

import { createMlp, serializeMlp, TrainerSession, type MlpConfig } from "../lib/mlp";

interface StartMessage {
  type: "start";
  config: MlpConfig;
  steps: number;
  batchSize: number;
  lr: number;
  seed: number;
}

let stopRequested = false;

const CHUNK = 10;

async function run(message: StartMessage) {
  const started = performance.now();
  const model = createMlp(message.config);
  const session = new TrainerSession(model, {
    lr: message.lr,
    batchSize: message.batchSize,
    seed: message.seed,
  });
  let reported = 0;
  const progressEvery = Math.max(1, Math.floor(message.steps / 120));

  for (let step = 0; step < message.steps; step += 1) {
    if (stopRequested) break;
    const loss = session.step();
    if (step - reported >= progressEvery || step === message.steps - 1) {
      reported = step;
      postMessage({
        type: "progress",
        step,
        loss,
        elapsedMs: performance.now() - started,
      });
    }
    if (step % CHUNK === 0) {
      // Yield so a "stop" message can be processed between chunks.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  const result = session.result();
  postMessage({
    type: "done",
    result,
    weights: serializeMlp(model),
    elapsedMs: performance.now() - started,
    stopped: stopRequested,
  });
}

self.onmessage = (event: MessageEvent<StartMessage | { type: "stop" }>) => {
  if (event.data.type === "stop") {
    stopRequested = true;
    return;
  }
  if (event.data.type === "start") {
    stopRequested = false;
    run(event.data).catch((error: unknown) => {
      postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
    });
  }
};
