import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { checkSecret } from "./identity.js";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageInfo = JSON.parse(fs.readFileSync(path.join(rootDir, "package.json"), "utf8")) as {
  version: string;
  stage?: string;
  description: string;
};

/** Everything this engine answers lives under this prefix on the home origin. */
export const BASE_PATH = "/faces";

const production = process.env.NODE_ENV === "production";
const gatewaySecret = process.env.GATEWAY_SECRET?.trim() || null;

// The gateway secret is the only thing standing between this server and anyone
// who can reach its port, so the image refuses to run without a good one; a
// source checkout without one is development and gets the developer identity.
const secretProblem = checkSecret(gatewaySecret, production);
if (secretProblem) throw new Error(secretProblem);

/** CPU threads the models may use. Left unbounded, onnxruntime runs the
 *  ResNet50 recogniser on every core and sharp's decode pool piles on; the
 *  default leaves one core for whatever else the box is doing. ORT_THREADS
 *  overrides (1 throttles hardest). */
function threadBudget(): number {
  const override = Number.parseInt(process.env.ORT_THREADS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  const cores = (typeof os.availableParallelism === "function" ? os.availableParallelism() : os.cpus().length) || 2;
  return Math.max(1, cores - 1);
}

/** Execution providers to try first, e.g. "cuda,cpu"; CPU is always the last resort. */
function providers(): string[] {
  const raw = (process.env.ORT_PROVIDERS ?? "").trim();
  const list = raw ? raw.split(",").map((p) => p.trim().toLowerCase()).filter(Boolean) : [];
  if (!list.includes("cpu")) list.push("cpu");
  return list;
}

export const config = {
  host: process.env.HOST ?? "127.0.0.1",
  port: Number(process.env.PORT ?? 4400),
  /** Where det_500m.onnx and w600k_r50.onnx are (the image: /app/models/face). */
  modelsDir: process.env.MODELS_PATH?.trim() || path.join(rootDir, "models", "face"),
  threads: threadBudget(),
  providers: providers(),
  /** Load the models on start rather than on the first photo, so the first scan
   *  of the night does not wait on it. */
  warm: /^(1|true|yes)$/i.test(process.env.WARM ?? ""),
  gatewaySecret,
  /** Development only: who the server pretends is signed in when no secret is set, "id:name". */
  devUser: process.env.DEV_USER ?? "dev:Developer",
  production,
  version: packageInfo.version,
  stage: packageInfo.stage ?? null
};
