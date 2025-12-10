FROM python:3.11-slim

# FFmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# uv 설치
RUN pip install uv

COPY pyproject.toml .
RUN uv sync

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "src/run.py"]
