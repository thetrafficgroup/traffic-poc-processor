# bookworm, not bullseye: Debian 11 left LTS in 2026-08 and its debian-security
# pool now 404s, which fails apt-get install (build 95532393, 2026-10-22).
FROM python:3.11.17-slim-bookworm

WORKDIR /

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "-u", "/handler.py"]