// Real detection: the native onnxruntime backend and the two models. Skipped
// when the recogniser has not been fetched (a fresh clone, CI) or the binding
// cannot load on this platform — the route tests cover the contract without it.
import { beforeAll, describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FaceDetector, cropFace } from "../src/arcface.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const modelsDir = path.join(here, "..", "models", "face");
let detector: FaceDetector | null = null;

beforeAll(async () => {
  const candidate = new FaceDetector({ modelsDir, threads: 2, providers: ["cpu"] });
  if (!candidate.modelsPresent()) {
    console.warn("face models not fetched, skipping detection test (npm run fetch:face-model)");
    return;
  }
  try {
    await candidate.warm();
    detector = candidate;
  } catch (err) {
    console.warn("face engine unavailable, skipping detection test:", err instanceof Error ? err.message : err);
  }
}, 120_000);

describe("face detection (ArcFace / onnxruntime)", () => {
  it("finds faces with 512-d unit vectors, normalised boxes and a crop each", async () => {
    if (!detector) return;
    const bytes = fs.readFileSync(path.join(here, "fixtures", "faces-sample.jpg"));
    const { image, faces } = await detector.detect(bytes);
    expect(faces.length).toBeGreaterThan(0);
    for (const face of faces) {
      expect(face.embedding.length).toBe(512);
      let norm = 0; for (const v of face.embedding) norm += v * v;
      expect(Math.sqrt(norm)).toBeCloseTo(1, 3);
      expect(face.box).toHaveLength(4);
      for (const v of face.box) { expect(Number.isFinite(v)).toBe(true); expect(v).toBeGreaterThan(-0.5); expect(v).toBeLessThan(1.5); }
      expect(face.score).toBeGreaterThan(0);
      const crop = await cropFace(image, face.box);
      expect(crop?.subarray(0, 4).toString("latin1")).toBe("RIFF");
    }
    const info = await detector.info();
    expect(info.providers).toContain("cpu");
  }, 120_000);
});
