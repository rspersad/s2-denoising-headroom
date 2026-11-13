FROM python:3.11-slim

LABEL org.opencontainers.image.title="praxis-cpu" \
      org.opencontainers.image.description="CPU stages for S2-SHIPS denoising/detection praxis" \
      org.opencontainers.image.version="0.2"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==2.4.6 \
        rasterio==1.4.4 \
        bm3d==4.0.3

RUN python -c "import numpy, rasterio, bm3d; print('imports ok; GDAL', rasterio.__gdal_version__)"

WORKDIR /work
