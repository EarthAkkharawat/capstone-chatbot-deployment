FROM python:3.10-slim-bullseye AS base

# Copy only the requirements file
COPY requirements.txt .

# Install build dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y openmpi-bin libopenmpi-dev

# Install PyTorch and dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Python requirements
RUN pip install --no-cache-dir pythainlp
RUN pip install --no-cache-dir --requirement requirements.txt
RUN python -m pip install --pre --extra-index-url https://pypi.nvidia.com optimum-nvidia

# Remove build dependencies
RUN apt-get purge -y --auto-remove gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*