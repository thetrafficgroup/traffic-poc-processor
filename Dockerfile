# Imagen base oficial con CUDA 11.8 y Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evitar interacciones en la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv python3.10-dev \
    git curl wget unzip libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Usar python3.10 como default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Crear carpeta de trabajo
WORKDIR /app

# Copiar los archivos del repositorio
COPY . .

# Instalar PyTorch, Torchvision, Torchaudio con CUDA 11.8
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118

# Instalar dependencias restantes del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint para RunPod (handler.handler → handler.py → def handler(event))
CMD ["python", "-m", "runpod"]