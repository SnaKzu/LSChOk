FROM nixos/nix:latest

# Install system dependencies
RUN nix-env -iA nixpkgs.python312 nixpkgs.gcc nixpkgs.ffmpeg nixpkgs.libGL nixpkgs.glib

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV OPENCV_IO_ENABLE_OPENEXR=0

WORKDIR /app/backend

# Create startup script
RUN echo '#!/bin/sh' > /start.sh && \
    echo 'cd /app/backend' >> /start.sh && \
    echo 'exec /opt/venv/bin/gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app_senas:app' >> /start.sh && \
    chmod +x /start.sh

ENTRYPOINT ["/bin/sh", "/start.sh"]
