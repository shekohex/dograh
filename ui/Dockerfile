FROM node:20-alpine AS deps
WORKDIR /app

RUN apk add --no-cache python3 make g++ libc6-compat

COPY ui/package.json ui/package-lock.json ./

RUN rm -rf node_modules && \
    npm ci || npm install

FROM node:20-alpine AS builder
WORKDIR /app

RUN apk add --no-cache libc6-compat

COPY --from=deps /app/node_modules ./node_modules

COPY ui/package.json ui/package-lock.json ./
COPY ui/tsconfig.json ./
COPY ui/next.config.ts ./
COPY ui/components.json ./
COPY ui/sentry.edge.config.ts ./
COPY ui/sentry.server.config.ts ./
COPY ui/postcss.config.mjs ./
COPY ui/public ./public
COPY ui/src ./src

ENV NEXT_PUBLIC_NODE_ENV="oss"
ENV NEXT_PUBLIC_AUTH_PROVIDER="local"
ENV NEXT_PUBLIC_DEPLOYMENT_MODE="oss"
ENV BACKEND_URL="http://api:8000"
ENV NEXT_TELEMETRY_DISABLED="1"
ENV NEXT_PUBLIC_CHATWOOT_URL="https://chat.dograh.com"
ENV NEXT_PUBLIC_CHATWOOT_TOKEN="3fkFx2mCEjNHjM9gaNc4A82X"

ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npm run build && \
    rm -rf /tmp/* /root/.npm /root/.next/cache

FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

EXPOSE 3010

CMD sh -c "echo '🚀 Application ready at http://localhost:3010' && PORT=3010 node server.js"
