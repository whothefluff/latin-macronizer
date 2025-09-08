# --- Stage 1 ---
# Use the full OS to compile everything and prepare our binaries.
FROM ubuntu:24.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Install build tools and binutils (for 'strip')
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libfl-dev unzip git wget ca-certificates binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build Morpheus
RUN git clone https://github.com/Alatius/morpheus.git && \
    cd morpheus/src && make && make install && cd .. && \
    ./update.sh && ./update.sh
# Build RFTagger
RUN wget -q https://www.cis.uni-muenchen.de/~schmid/tools/RFTagger/data/RFTagger.zip && \
    unzip RFTagger.zip && \
    cd RFTagger/src && make && make install

# Strip binaries to remove debug symbols and reduce size
RUN strip /build/morpheus/bin/cruncher /usr/local/bin/rft-annotate /usr/local/bin/rft-train


# --- Stage 2: Final Runtime ---
# Start from the same base, but be extremely careful about what we add.
FROM ubuntu:24.04 AS final

LABEL description="Optimized Ubuntu runtime for Latin Macronizer."

# Install ONLY the necessary runtime dependencies and clean up in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    git \
    libfl2 \
    ca-certificates \
    # Clean up the apt cache to save space
    && rm -rf /var/lib/apt/lists/*
# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 -m -s /bin/bash appuser

WORKDIR /app
# Grant ownership of the app directory to the user
RUN chown -R appuser:appgroup /app
# Copy the *stripped* binaries and data from the builder stage
COPY --from=builder --chown=appuser:appgroup /build/morpheus/bin /app/morpheus/bin
COPY --from=builder --chown=appuser:appgroup /build/morpheus/stemlib /app/morpheus/stemlib
COPY --from=builder /usr/local/bin/rft-annotate /usr/local/bin/
COPY --from=builder /usr/local/bin/rft-train /usr/local/bin/
# Install Python packages.
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir debugpy && \
    # Ensure the venv is also owned by the appuser
    chown -R appuser:appgroup /opt/venv
# Copy application code
COPY --chown=appuser:appgroup . .
# Ensure the training script is executable
RUN chmod +x ./train-rftagger.sh
# Switch to the non-root user for all subsequent commands
USER appuser
# Make tools directly accessible from the command line
ENV PATH="/opt/venv/bin:/app/morpheus/bin:$PATH"
# The default is to start a shell for interactive training.
CMD ["/bin/bash"]