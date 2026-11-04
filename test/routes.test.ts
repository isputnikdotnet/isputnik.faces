// The contract, with a stub in place of the models: who may call what, the
// shapes that come back, and the three ways a detect call fails.
import { afterEach, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import { buildApp, type Detector } from "../src/app.js";
import { DecodeError, FACE_EMBEDDING_MODEL, ModelsMissingError, type DecodedImage, type DetectedFace } from "../src/arcface.js";
import { createLogBuffer } from "../src/log-buffer.js";

const SECRET = "faces-secret-for-tests-0123456789abcdef";

const admin = {
  "x-isputnik-gateway": SECRET,
  "x-isputnik-user-id": "system:home",
  "x-isputnik-user-name": "isputnik.home",
  "x-isputnik-user-role": "admin",
  "x-isputnik-request-id": "home-abc"
};
const member = { ...admin, "x-isputnik-user-id": "mum", "x-isputnik-user-name": "Mum", "x-isputnik-user-role": "member" };

/** A 4×4 opaque RGB image, as a decode would give. */
const image: DecodedImage = { rgb: Buffer.alloc(4 * 4 * 3, 128), width: 4, height: 4 };
const face: DetectedFace = { box: [0.1, 0.2, 0.3, 0.4], score: 0.9, embedding: new Float32Array(512).fill(1 / Math.sqrt(512)) };

function stub(behaviour: "faces" | "none" | "missing" | "undecodable" = "faces"): Detector & { calls: number } {
  return {
    calls: 0,
    modelsPresent: () => behaviour !== "missing",
    loaded: () => behaviour === "faces",
    info: async () => ({ model: FACE_EMBEDDING_MODEL, providers: ["cpu"], gpu: false }),
    warm: async () => undefined,
    async detect() {
      this.calls += 1;
      if (behaviour === "missing") throw new ModelsMissingError("no models");
      if (behaviour === "undecodable") throw new DecodeError("Input buffer contains unsupported image format");
      return { image, faces: behaviour === "faces" ? [face] : [] };
    }
  };
}

let app: FastifyInstance | null = null;

async function boot(detector: Detector = stub(), secret: string | null = SECRET) {
  app = await buildApp({ gatewaySecret: secret, devUser: secret ? null : "dev:Developer", detector, version: "0.1.0-test", stage: "alpha", logBuffer: createLogBuffer(50, { write: () => undefined }) });
  return app;
}

afterEach(async () => {
  await app?.close();
  app = null;
});

describe("health and the signed probe", () => {
  it("answers health to anyone, naming the model", async () => {
    const res = await (await boot()).inject({ method: "GET", url: "/faces/api/health" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ ok: true, version: "0.1.0-test", model: FACE_EMBEDDING_MODEL, modelsPresent: true, loaded: true, providers: ["cpu"], gpu: false });
  });

  it("refuses /me without the secret and answers it with", async () => {
    const server = await boot();
    expect((await server.inject({ method: "GET", url: "/faces/api/me" })).statusCode).toBe(401);
    expect((await server.inject({ method: "GET", url: "/faces/api/me", headers: { ...admin, "x-isputnik-gateway": "wrong" } })).statusCode).toBe(401);
    const res = await server.inject({ method: "GET", url: "/faces/api/me", headers: member });
    expect(res.statusCode).toBe(200);
    expect(res.json().user).toEqual({ id: "mum", name: "Mum", role: "member" });
  });

  it("records a refused caller as an event the admin can read", async () => {
    const server = await boot();
    await server.inject({ method: "GET", url: "/faces/api/me" });
    const res = await server.inject({ method: "GET", url: "/faces/api/admin/events", headers: admin });
    expect(res.statusCode).toBe(200);
    const events = res.json().events as { event: string; detail: string }[];
    expect(events[0]).toMatchObject({ event: "gateway.rejected", detail: "GET /faces/api/me: no gateway header" });
    expect(res.json().rejected24h).toBe(1);
  });
});

describe("POST /faces/api/detect", () => {
  it("is for admins only", async () => {
    const server = await boot();
    const res = await server.inject({ method: "POST", url: "/faces/api/detect", headers: { ...member, "content-type": "image/jpeg" }, payload: Buffer.from("x") });
    expect(res.statusCode).toBe(403);
  });

  it("answers boxes, base64 vectors and crops, one call per photo", async () => {
    const detector = stub();
    const server = await boot(detector);
    const res = await server.inject({ method: "POST", url: "/faces/api/detect", headers: { ...admin, "content-type": "image/jpeg" }, payload: Buffer.from("jpeg-bytes") });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.model).toBe(FACE_EMBEDDING_MODEL);
    expect(body.width).toBe(4);
    expect(body.faces).toHaveLength(1);
    expect(body.faces[0].box).toEqual([0.1, 0.2, 0.3, 0.4]);
    const vector = Buffer.from(body.faces[0].embedding, "base64");
    expect(vector.length).toBe(512 * 4);
    expect(new Float32Array(vector.buffer, vector.byteOffset, 512)[0]).toBeCloseTo(1 / Math.sqrt(512));
    // The stub's 4×4 image crops to a real WebP.
    expect(typeof body.faces[0].crop).toBe("string");
    expect(Buffer.from(body.faces[0].crop, "base64").subarray(0, 4).toString("latin1")).toBe("RIFF");
    expect(detector.calls).toBe(1);
  });

  it("skips the crops when asked", async () => {
    const server = await boot();
    const res = await server.inject({ method: "POST", url: "/faces/api/detect?crops=0", headers: { ...admin, "content-type": "image/jpeg" }, payload: Buffer.from("jpeg-bytes") });
    expect(res.json().faces[0].crop).toBeNull();
  });

  it("refuses a body that is not an image", async () => {
    const server = await boot();
    const res = await server.inject({ method: "POST", url: "/faces/api/detect", headers: { ...admin, "content-type": "application/json" }, payload: "{}" });
    expect(res.statusCode).toBe(415);
  });

  it("says 422 for a photo it cannot decode, and 503 while the models are missing", async () => {
    const bad = await boot(stub("undecodable"));
    expect((await bad.inject({ method: "POST", url: "/faces/api/detect", headers: { ...admin, "content-type": "image/png" }, payload: Buffer.from("nope") })).statusCode).toBe(422);
    await bad.close();
    const missing = await boot(stub("missing"));
    const res = await missing.inject({ method: "POST", url: "/faces/api/detect", headers: { ...admin, "content-type": "image/png" }, payload: Buffer.from("nope") });
    expect(res.statusCode).toBe(503);
    expect((await missing.inject({ method: "GET", url: "/faces/api/health" })).json().modelsPresent).toBe(false);
  });

  it("runs as the developer when no secret is set", async () => {
    const server = await boot(stub("none"), null);
    const res = await server.inject({ method: "POST", url: "/faces/api/detect", headers: { "content-type": "image/jpeg" }, payload: Buffer.from("x") });
    expect(res.statusCode).toBe(200);
    expect(res.json().faces).toEqual([]);
  });
});

describe("the admin's window", () => {
  it("hands over the log tail and an empty backup", async () => {
    const server = await boot();
    const log = await server.inject({ method: "GET", url: "/faces/api/admin/log?lines=10", headers: admin });
    expect(log.statusCode).toBe(200);
    expect(log.json().available).toBe(true);
    expect((await server.inject({ method: "GET", url: "/faces/api/admin/backup", headers: admin })).statusCode).toBe(204);
    expect((await server.inject({ method: "GET", url: "/faces/api/admin/log", headers: member })).statusCode).toBe(403);
  });
});
