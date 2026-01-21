FROM python:3.12-slim AS deps

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY api/requirements.txt .
RUN pip install -r requirements.txt

COPY pipecat /tmp/pipecat
RUN pip install '/tmp/pipecat[cartesia,deepgram,openai,elevenlabs,groq,google,azure,sarvam,soundfile,silero,webrtc,local-smart-turn-v3,speechmatics]' && rm -rf /tmp/pipecat

RUN find /opt/venv -type f -name '*.pyc' -delete && \
    find /opt/venv -type f -name '*.pyo' -delete && \
    find /opt/venv -type d -name '__pycache__' -delete

FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY --from=deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY api ./api
COPY scripts/start_services.sh ./scripts/start_services.sh

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["bash", "-c", "./scripts/start_services.sh && tail -f ./logs/latest/*.log"]
