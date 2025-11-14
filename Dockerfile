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
ENV PYTHONPATH="/app"

WORKDIR /app

ENTRYPOINT ["/opt/venv/bin/gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "--chdir", "/app/backend", "app_senas:app"]
