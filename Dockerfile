# ── Stage 1: install all deps ─────────────────────────────────────
# sharp and onnxruntime-node ship prebuilt binaries for linux/x64; the build
# tools are here for the day a prebuild is missing for the image's Node.
FROM node:26-slim AS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci

# ── Stage 2: build the server ─────────────────────────────────────
FROM deps AS build
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# ── Stage 3: runtime-only node_modules ────────────────────────────
FROM deps AS prod-deps
RUN npm ci --omit=dev
# onnxruntime-node ships binaries for every OS and arch in one package (~220 MB
# of foreign-platform dead weight). Keep only this image's platform. Must happen
# HERE, not in the final stage — a later RUN rm can't shrink an earlier COPY layer.
RUN rm -rf node_modules/onnxruntime-node/bin/napi-v6/darwin \
           node_modules/onnxruntime-node/bin/napi-v6/win32 \
    && find node_modules/onnxruntime-node/bin/napi-v6/linux \
            -mindepth 1 -maxdepth 1 -type d ! -name "$(node -p 'process.arch')" \
            -exec rm -rf {} +

# ── Stage 4: the recogniser ───────────────────────────────────────
# The 167 MB ArcFace model is fetched from a GitHub release rather than carried in
# git. Its own stage keeps the download out of the code layers — a server change
# doesn't refetch it — and the script verifies a pinned SHA-256, so a replaced or
# truncated asset fails the build instead of shipping an engine whose vectors are
# quietly wrong.
FROM node:26-slim AS face-model
WORKDIR /build
COPY scripts/fetch-face-model.mjs ./scripts/
RUN node scripts/fetch-face-model.mjs --dest /build/models/face

# ── Stage 5: production image ─────────────────────────────────────
FROM node:26-slim
WORKDIR /app

# Unraid draws the container's icon from this label when its saved template
# has none, so a template edit or an update cannot leave the tile blank.
LABEL net.unraid.docker.icon="https://raw.githubusercontent.com/isputnikdotnet/isputnik.faces/main/Assets/icon.png"

# gosu drops the entrypoint from root to the runtime user (scripts/docker-entrypoint.sh).
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

COPY --from=prod-deps /build/node_modules ./node_modules
COPY --from=build /build/dist ./dist
# Carries "type": "module", and config.ts reads the version from it (one level
# above dist/).
COPY package.json ./

# The two models: the 2.5 MB detector from git, the recogniser from the stage above.
COPY models/face/det_500m.onnx ./models/face/
COPY --from=face-model /build/models/face/w600k_r50.onnx ./models/face/

# Import smoke test: every production dependency must load from this tree, so a
# missing package — or a pruned-away native binding — fails THIS build instead
# of the container at startup.
RUN node -e 'const deps=Object.keys(require("./package.json").dependencies);(async()=>{for(const d of deps){try{await import(d)}catch(e){console.error("runtime dep failed to load:",d,"-",e.message);process.exit(1)}}console.log(deps.length+" runtime deps load OK")})()'

ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=4400
ENV MODELS_PATH=/app/models/face
# GATEWAY_SECRET must be set at run time; the server refuses to start without it.
# Who the server runs as. Nothing is written, so any user will do.
ENV PUID=1000
ENV PGID=1000

EXPOSE 4400

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||4400)+'/faces/api/health').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"

COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

CMD ["node", "dist/index.js"]
