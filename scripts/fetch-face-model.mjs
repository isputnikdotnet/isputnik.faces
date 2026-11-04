// Fetches the ArcFace recogniser (InsightFace buffalo_l / w600k_r50) into
// models/face/.
//
// The file is 167 MB and lived in git-LFS until every image build pulling it
// exhausted the repository's LFS budget — the checkout failed before anything
// compiled, so no image was published at all. It now lives on a GitHub release
// (a plain asset download, outside the LFS quota) and is fetched here: during
// the Docker build, and by developers running the engine from a source checkout.
//
// The small detector (det_500m.onnx, 2.5 MB) stays in git — it costs nothing to
// carry and keeps the models directory non-empty in a fresh clone.
//
// Usage:  node scripts/fetch-face-model.mjs [--dest <dir>] [--force]
// Env:    FACE_MODEL_URL overrides the download source (air-gapped builds, mirrors).

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";

const FILE_NAME = "w600k_r50.onnx";
// Pinned by content, not by version: the URL is a mutable pointer (a release
// asset can be replaced) while this digest is what the recogniser was validated
// against. A mismatch means the file is not the model this code expects, so the
// build fails rather than shipping an image that embeds subtly wrong vectors —
// which would not crash, it would silently cluster faces wrong.
const SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43";
const DEFAULT_URL =
  "https://github.com/isputnikdotnet/isputnik.home/releases/download/face-model-buffalo-l-v1/w600k_r50.onnx";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const args = { dest: path.join(repoRoot, "models", "face"), force: false };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--dest") {
      const value = argv[i + 1];
      if (!value) throw new Error("--dest needs a directory");
      args.dest = path.resolve(value);
      i += 1;
    } else if (argv[i] === "--force") {
      args.force = true;
    } else {
      throw new Error(`unknown argument: ${argv[i]}`);
    }
  }
  return args;
}

async function sha256File(file) {
  const hash = crypto.createHash("sha256");
  await pipeline(fs.createReadStream(file), hash);
  return hash.digest("hex");
}

function human(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function main() {
  const { dest, force } = parseArgs(process.argv.slice(2));
  const target = path.join(dest, FILE_NAME);
  const url = process.env.FACE_MODEL_URL || DEFAULT_URL;

  if (fs.existsSync(target) && !force) {
    // Verify rather than trust the path: a half-written file from an interrupted
    // run is the case that would otherwise fail much later, inside onnxruntime.
    const have = await sha256File(target);
    if (have === SHA256) {
      console.log(`face model already present and verified: ${target}`);
      return;
    }
    console.log(`face model at ${target} has unexpected digest ${have.slice(0, 12)}… — refetching`);
  }

  fs.mkdirSync(dest, { recursive: true });
  console.log(`fetching ${FILE_NAME} from ${url}`);

  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`download failed: ${response.status} ${response.statusText}`);
  }

  // Download to a sibling temp file and rename only after the digest checks out,
  // so an interrupted or corrupted fetch can never leave something at the real
  // path that a later run (or the Docker COPY) would treat as the model.
  const temp = `${target}.partial`;
  await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(temp));

  const digest = await sha256File(temp);
  if (digest !== SHA256) {
    fs.rmSync(temp, { force: true });
    throw new Error(`checksum mismatch: expected ${SHA256}, got ${digest}`);
  }

  fs.renameSync(temp, target);
  console.log(`face model ready: ${target} (${human(fs.statSync(target).size)}, sha256 verified)`);
}

// Set exitCode rather than calling process.exit: a forced exit while the fetch
// body or file stream is still closing aborts libuv on Windows (UV_HANDLE_CLOSING
// assertion) and reports 127, hiding the real failure behind a crash.
main().catch((error) => {
  console.error(`fetch-face-model: ${error.message}`);
  process.exitCode = 1;
});
