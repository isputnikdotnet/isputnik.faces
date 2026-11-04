// The face engine: no pages, no database. Home sends a photo, this server
// answers with the faces in it — where each is, a vector that says who it looks
// like, and a small crop for the person's avatar — and remembers nothing
// (home's docs/face-engine-plan.md). Reached only by home's server, signed with
// the shared secret; the browser never gets past home's gateway to /detect.
import fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest, type FastifyServerOptions } from "fastify";
import { z } from "zod";
import { BASE_PATH } from "./config.js";
import { DecodeError, FACE_EMBEDDING_MODEL, ModelsMissingError, cropFace, type DecodedImage, type DetectedFace, type EngineInfo } from "./arcface.js";
import { EVENT_LEVELS, EVENT_RETENTION_DAYS, EventLog, type EventLevel } from "./events.js";
import { clientIpFrom, devIdentity, forwardedRequestId, identityFrom, rejectionReason, type Identity } from "./identity.js";
import type { LogBuffer } from "./log-buffer.js";

declare module "fastify" {
  interface FastifyRequest {
    identity: Identity | null;
    /** The browser's address as the gateway saw it; the socket's otherwise. */
    clientIp: string;
  }
}

/** What the routes need of the models: the real FaceDetector, or a stub in tests. */
export interface Detector {
  modelsPresent(): boolean;
  loaded(): boolean;
  info(): Promise<EngineInfo>;
  warm(): Promise<void>;
  detect(input: Buffer): Promise<{ image: DecodedImage; faces: DetectedFace[] }>;
}

export interface AppOptions {
  gatewaySecret: string | null;
  /** Used when no secret is configured — development only. */
  devUser: string | null;
  detector: Detector;
  version: string;
  stage: string | null;
  logBuffer?: LogBuffer;
  fastify?: FastifyServerOptions;
}

const API = `${BASE_PATH}/api`;

/** A photo as it came off the camera; anything bigger is not a photo to find faces in. */
export const DETECT_BODY_LIMIT = 64 * 1024 * 1024;
/** Everything else is small JSON or nothing. */
const BODY_LIMIT = 64 * 1024;
const LOG_LINES_MAX = 500;

/** What sharp opens. HEIC is left out on purpose: the prebuilt libvips has no
 *  licence to decode it, and home converts such photos before sending. */
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff", "image/gif", "image/avif", "image/bmp"];

const eventsQuerySchema = z.object({
  level: z.enum(EVENT_LEVELS as [EventLevel, ...EventLevel[]]).optional(),
  event: z.string().trim().max(64).optional(),
  page: z.coerce.number().int().min(1).default(1),
  pageSize: z.coerce.number().int().min(10).max(100).default(25)
});

const logQuerySchema = z.object({
  lines: z.coerce.number().int().min(10).max(LOG_LINES_MAX).default(200)
});

const detectQuerySchema = z.object({
  /** 0 skips the avatars, for a caller that only wants the vectors. */
  crops: z.enum(["0", "1"]).default("1")
});

export interface DetectAnswer {
  model: string;
  width: number;
  height: number;
  faces: { box: [number, number, number, number]; score: number; embedding: string; crop: string | null }[];
}

export async function buildApp(options: AppOptions): Promise<FastifyInstance> {
  const secret = options.gatewaySecret;
  const dev = options.devUser && !secret ? devIdentity(options.devUser) : null;
  const events = new EventLog();
  const { detector } = options;

  const app = fastify({
    bodyLimit: BODY_LIMIT,
    trustProxy: false,
    ...options.fastify,
    // Every line names who asked and from where, and carries home's request id
    // when the request came through the gateway, so the two logs line up.
    genReqId: (raw) => forwardedRequestId(raw.headers, secret) ?? `f-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    childLoggerFactory(logger, bindings, loggerOptions, raw) {
      const identity = secret ? identityFrom(raw.headers, secret) : dev;
      return logger.child({ ...bindings, user: identity?.id ?? null, ip: clientIpFrom(raw.headers, secret, raw.socket?.remoteAddress) }, loggerOptions);
    }
  });

  // The photo itself, as bytes. Only on /detect: the limit there is the photo's.
  for (const type of IMAGE_TYPES) {
    app.addContentTypeParser(type, { parseAs: "buffer", bodyLimit: DETECT_BODY_LIMIT }, (_request, body, done) => done(null, body));
  }
  // Anything else sent as a body is refused by the parser, not read.
  app.addContentTypeParser("*", (_request, _payload, done) => done(null, undefined));

  app.decorateRequest("identity", null);
  app.decorateRequest("clientIp", "");
  app.addHook("onRequest", async (request) => {
    request.identity = secret ? identityFrom(request.headers, secret) : dev;
    request.clientIp = clientIpFrom(request.headers, secret, request.ip);
  });

  const note = (request: FastifyRequest | null, level: EventLevel, event: string, detail: string) => {
    events.record({
      level,
      event,
      actorId: request?.identity?.id ?? null,
      actorName: request?.identity?.name ?? null,
      ip: request?.clientIp ?? null,
      requestId: request ? String(request.id) : null,
      detail
    });
  };

  const requireIdentity = async (request: FastifyRequest, reply: FastifyReply) => {
    if (request.identity) return;
    note(request, "warn", "gateway.rejected", `${request.method} ${request.url.split("?")[0]}: ${rejectionReason(request.headers, secret)}`);
    return reply.code(401).send({ error: "Not signed in" });
  };

  const requireAdmin = async (request: FastifyRequest, reply: FastifyReply) => {
    await requireIdentity(request, reply);
    if (reply.sent) return;
    if (request.identity!.role !== "admin") return reply.code(403).send({ error: "Admins only" });
  };

  // ── Health and the signed probe ─────────────────────────────────────────

  app.get(`${API}/health`, async () => {
    const loaded = detector.loaded();
    const info = loaded ? await detector.info().catch(() => null) : null;
    return {
      ok: true,
      version: options.version,
      stage: options.stage,
      model: FACE_EMBEDDING_MODEL,
      modelsPresent: detector.modelsPresent(),
      loaded,
      providers: info?.providers ?? null,
      gpu: info?.gpu ?? null
    };
  });

  // Who home says this is: the signed probe the Dashboard uses to tell a wrong
  // secret from a working one.
  app.get(`${API}/me`, { preHandler: requireIdentity }, async (request) => {
    const identity = request.identity!;
    return {
      user: { id: identity.id, name: identity.name, role: identity.role },
      version: options.version,
      stage: options.stage,
      model: FACE_EMBEDDING_MODEL
    };
  });

  // ── The work ────────────────────────────────────────────────────────────

  // One photo at a time: two inferences at once compete for the same cores and
  // finish no sooner, and two sharp pipelines over one bad file have killed a
  // process outright before (home's CLAUDE.md). Callers wait their turn.
  let turn: Promise<unknown> = Promise.resolve();
  const inTurn = <T>(work: () => Promise<T>): Promise<T> => {
    const next = turn.then(work, work);
    turn = next.catch(() => undefined);
    return next;
  };

  app.post(`${API}/detect`, { onRequest: requireAdmin, bodyLimit: DETECT_BODY_LIMIT }, async (request, reply) => {
    if (!Buffer.isBuffer(request.body)) {
      return reply.code(415).send({ error: `Send the photo itself, as one of: ${IMAGE_TYPES.join(", ")}.` });
    }
    const query = detectQuerySchema.parse(request.query ?? {});
    const bytes = request.body;
    const started = performance.now();
    let result: { image: DecodedImage; faces: DetectedFace[] };
    try {
      result = await inTurn(() => detector.detect(bytes));
    } catch (err) {
      if (err instanceof ModelsMissingError) {
        note(request, "error", "engine.models-missing", err.message);
        return reply.code(503).send({ error: err.message });
      }
      if (err instanceof DecodeError) {
        note(request, "warn", "detect.undecodable", `${bytes.length} bytes: ${err.message}`);
        return reply.code(422).send({ error: `The photo could not be decoded: ${err.message}` });
      }
      note(request, "error", "detect.failed", err instanceof Error ? err.message : String(err));
      throw err;
    }
    const faces: DetectAnswer["faces"] = [];
    for (const face of result.faces) {
      const crop = query.crops === "1" ? await cropFace(result.image, face.box) : null;
      faces.push({
        box: face.box,
        score: face.score,
        embedding: Buffer.from(face.embedding.buffer, face.embedding.byteOffset, face.embedding.byteLength).toString("base64"),
        crop: crop ? crop.toString("base64") : null
      });
    }
    request.log.info({ faces: faces.length, bytes: bytes.length, ms: Math.round(performance.now() - started) }, "detected");
    const answer: DetectAnswer = { model: FACE_EMBEDDING_MODEL, width: result.image.width, height: result.image.height, faces };
    return answer;
  });

  // ── The admin's window ──────────────────────────────────────────────────

  app.get(`${API}/admin/events`, { preHandler: requireAdmin }, async (request) => {
    const query = eventsQuerySchema.parse(request.query ?? {});
    return {
      ...events.list({ level: query.level ?? null, event: query.event || null, page: query.page, pageSize: query.pageSize }),
      retentionDays: EVENT_RETENTION_DAYS,
      rejected24h: events.countSince("gateway.rejected", 24)
    };
  });

  app.get(`${API}/admin/log`, { preHandler: requireAdmin }, async (request) => {
    const query = logQuerySchema.parse(request.query ?? {});
    return {
      lines: options.logBuffer?.tail(query.lines) ?? [],
      capacity: options.logBuffer?.capacity ?? 0,
      available: Boolean(options.logBuffer)
    };
  });

  // Nothing to hand over: this engine keeps no data. Home's registry knows that
  // and never asks; an older home that does gets an empty answer, not an error.
  app.get(`${API}/admin/backup`, { preHandler: requireAdmin }, async (_request, reply) => reply.code(204).send());

  app.setErrorHandler((err: unknown, request, reply) => {
    if (err instanceof z.ZodError) return reply.code(400).send({ error: "Invalid request", details: err.issues });
    const code = (err as { statusCode?: unknown }).statusCode;
    const status = typeof code === "number" && code >= 400 ? code : 500;
    if (status >= 500) request.log.error({ err }, "request failed");
    return reply.code(status).send({ error: status >= 500 ? "Something went wrong" : err instanceof Error ? err.message : String(err) });
  });

  app.decorate("events", events);
  note(null, "info", "engine.started", `Face engine ${options.version} started${secret ? "" : " without a gateway secret (development)"}.`);
  return app;
}

declare module "fastify" {
  interface FastifyInstance {
    events: EventLog;
  }
}
