# isputnik.faces

**An add-on for [isputnik.home](https://github.com/isputnikdotnet/isputnik.home):**
its face recognition engine. It finds the faces in a photo and describes each
one, behind the home server, which keeps every face, name and person and draws
every screen. It requires the isputnik.home container, is installed beside it
and is reached only through home's `/faces` gateway; this server has no pages
and **no data**: a photo goes in, its faces come out, nothing is remembered.

Why it is a container of its own: the ONNX runtime and the two InsightFace
models are ~450 MB that change about never, and home releases every few days.
Here they are pulled once. The design is home's `docs/face-engine-plan.md`.

Status: 0.1.0, alpha. CPU only; a CUDA image is planned.

## The contract, from this side

- Reached only by home's server, signed with `GATEWAY_SECRET` in `X-Isputnik-*`
  headers (home's `docs/apps-platform-plan.md` → The contract); without the
  secret a request is anonymous. Home's gateway forwards only the admin window
  to browsers; `/detect` is called by home's own scanner.
- The model id (`buffalo_l/w600k_r50`) is the one home stamps on its face rows,
  so an install that switches from in-process recognition re-embeds nothing.
- The engine never calls home.

| Route | Who | What |
|---|---|---|
| `GET /faces/api/health` | anyone | `{ ok, version, model, modelsPresent, loaded, providers, gpu }` |
| `GET /faces/api/me` | signed | who home says this is — the Dashboard's signed probe |
| `POST /faces/api/detect?crops=1` | admin | body: the photo's bytes (`image/jpeg`, `png`, `webp`, `tiff`, `gif`, `avif`, `bmp`; up to 64 MB). Answer: `{ model, width, height, faces: [{ box: [x, y, w, h], score, embedding, crop }] }` — boxes normalised 0..1 of the upright image, `embedding` 512 float32 as base64, `crop` a 160 px WebP as base64 (or null with `crops=0`). 415 for a body that is not an image, 422 for one that will not decode, 503 while the recogniser is missing |
| `GET /faces/api/admin/events`, `/log` | admin | the admin's window, kept in memory |
| `GET /faces/api/admin/backup` | admin | 204 — nothing to hand over |

## Running it

```
docker network create isputnik          # once, shared with isputnik.home
docker run -d --name isputnik-faces --network isputnik \
  -e GATEWAY_SECRET=<the secret from Control panel → Settings › Apps → Faces> \
  ghcr.io/isputnikdotnet/isputnik.faces:latest
```

Then in isputnik.home: Control panel → Settings › Apps → Faces, address
`http://isputnik-faces:4400` (or `http://<host IP>:4400` with the port published,
`http://<its IP>:4400` on br0 with a fixed IP), paste the same secret, **Test
connection**. The image is `ghcr.io/isputnikdotnet/isputnik.faces` and, the same
bytes, `isputnikdotnet/isputnik.faces` on Docker Hub; every release tag publishes
both.

**Unraid** — the repository carries a Community Applications template,
[`isputnik-faces.xml`](isputnik-faces.xml), listed in CA as **isputnik-faces**. It
publishes port 4400 on `bridge` (safe: without the secret every request is
answered 401), mounts nothing, and asks for the gateway secret, which is made in
home first because the field is required.

| Variable | Default | What |
|---|---|---|
| `GATEWAY_SECRET` | — | required in the image, at least 32 characters |
| `ORT_THREADS` | cores − 1 | CPU threads for the models and the decoder |
| `ORT_PROVIDERS` | `cpu` | execution providers to try first (`cuda,cpu`); CPU is always the last resort |
| `WARM` | `0` | `1` loads the models on start |
| `LOG_LEVEL` | `info` | pino level |
| `PUID` / `PGID` | 1000 | who it runs as |

## Developing

```
npm install
npm run fetch:face-model     # the 167 MB recogniser, verified by SHA-256
npm run dev                  # :4400, no secret = the developer identity
npm test                     # the route tests; detection too once the model is there
```

To point a home checkout at it: `FACES_URL=http://127.0.0.1:4400` and the same
`FACES_GATEWAY_SECRET` / `GATEWAY_SECRET` on both.

## Licence

AGPL-3.0-only, like isputnik.home. The models are InsightFace's
(`buffalo_l`: SCRFD-500MF and the w600k_r50 ArcFace recogniser), under their
own terms.
