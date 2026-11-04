import { config } from "./config.js";
import { buildApp } from "./app.js";
import { FaceDetector } from "./arcface.js";
import { createLogBuffer } from "./log-buffer.js";

// Every log line goes to stdout, and the newest few hundred stay in memory for
// the admin to read from home (Control panel → Overview › Logs).
const logBuffer = createLogBuffer(500);

const detector = new FaceDetector({
  modelsDir: config.modelsDir,
  threads: config.threads,
  providers: config.providers,
  log: {
    info: (message) => app.log.info(`face engine: ${message}`),
    warn: (message) => app.log.warn(`face engine: ${message}`)
  }
});

const app = await buildApp({
  gatewaySecret: config.gatewaySecret,
  devUser: config.production ? null : config.devUser,
  detector,
  version: config.version,
  stage: config.stage,
  logBuffer,
  fastify: {
    logger: { level: process.env.LOG_LEVEL?.trim() || "info", stream: logBuffer }
  }
});

if (!config.gatewaySecret) {
  app.log.warn(`No GATEWAY_SECRET: every request is treated as the developer identity (${config.devUser}). Never run like this where anyone else can reach the port.`);
}
if (!detector.modelsPresent()) {
  app.log.error(`Face recognition models not found in ${config.modelsDir}: run "npm run fetch:face-model". Detection answers 503 until they are there.`);
}

await app.listen({ host: config.host, port: config.port });
app.log.info(`Face engine ${config.version}: models at ${config.modelsDir}, ${config.threads} thread(s), providers ${config.providers.join(", ")}`);

if (config.warm && detector.modelsPresent()) {
  detector.warm().catch((err) => app.log.error({ err }, "face engine: the models did not load"));
}

// Orderly shutdown: Node is PID 1 in the container, so SIGTERM arrives here.
let shuttingDown = false;
async function shutdown(reason: string): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  app.log.info(`${reason} — shutting down`);
  setTimeout(() => process.exit(1), 10_000).unref();
  try {
    await app.close();
  } catch (err) {
    app.log.error({ err }, "Error while shutting down");
  }
  process.exit(0);
}
process.on("SIGTERM", () => void shutdown("SIGTERM received"));
process.on("SIGINT", () => void shutdown("SIGINT received"));
