FROM nixos/nix:latest

# Install system dependencies
RUN nix-env -iA nixpkgs.python312 nixpkgs.gcc nixpkgs.ffmpeg nixpkgs.libGL nixpkgs.glib nixpkgs.stdenv.cc.cc.lib

# Find and set library paths
RUN STDCPP_LIB=$(find /nix/store -name "libstdc++.so.6" | head -1 | xargs dirname) && \
    echo "export LD_LIBRARY_PATH=$STDCPP_LIB:\$LD_LIBRARY_PATH" >> /etc/profile && \
    echo "$STDCPP_LIB" > /etc/ld.so.conf.d/nix.conf && \
    ldconfig 2>/dev/null || true

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
ENV LD_LIBRARY_PATH="/nix/store/*-gcc-*/lib:/nix/store/*-glibc-*/lib:$LD_LIBRARY_PATH"

WORKDIR /app

ENTRYPOINT ["/opt/venv/bin/gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:8080", "--chdir", "/app/backend", "app_senas:app"]
