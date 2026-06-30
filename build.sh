#!/usr/bin/env bash
set -o errexit

apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    || true

pip install --upgrade pip
pip install -r requirements.txt

