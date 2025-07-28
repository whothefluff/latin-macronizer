# Start with a standard Debian/Ubuntu base
FROM ubuntu:24.04

# Prevent interactive prompts during apt installation
ARG DEBIAN_FRONTEND=noninteractive

# Set labels for identifying the image's purpose and stage
LABEL stage="golden-base"
LABEL description="Contains all compiled dependencies for the Latin Macronizer. Ready for training RFTagger."

# Set the working directory early
WORKDIR /app

# --- Block 1: Install System & Python Dependencies in a single layer ---
# Combines apt, venv creation, and pip installs.
# We also pre-install packages you'll likely need for your API.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfl-dev \
    unzip \
    git \
    wget \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    # Now, set up the Python virtual environment and install packages into it
    && python3 -m venv /opt/venv \
    # Note: We activate the venv for this RUN command only to install packages
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir debugpy

# Add the venv to the PATH for all subsequent commands and for the final container
ENV PATH="/opt/venv/bin:$PATH"

# --- Block 2: Create the non-root user and group ---
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 -m -s /bin/bash appuser && \
    # Grant ownership of the app directory now, before we copy files into it
    chown -R appuser:appgroup /app

# --- Block 3: Download, compile, and clean up Morpheus ---
# This entire operation is one logical step, so it belongs in one layer.
RUN git clone https://github.com/Alatius/morpheus.git && \
    cd morpheus/src && make && make install && cd .. && \
    ./update.sh && \
    ./update.sh && \
    cd ..

# --- Block 4: Download, compile, and clean up RFTagger ---
# Same logic as Morpheus: build, install, then clean up the build artifacts.
RUN wget -q https://www.cis.uni-muenchen.de/~schmid/tools/RFTagger/data/RFTagger.zip && \
    unzip RFTagger.zip && \
    cd RFTagger/src && make && make install && cd ../.. && \
    rm -rf RFTagger RFTagger.zip

# --- Block 5: Copy application code and set permissions ---
# Use the --chown flag copy the application source code into the container.
COPY --chown=appuser:appgroup . .

# Ensure the training script is executable
RUN chmod +x ./train-rftagger.sh

# Switch to the non-root user. All subsequent commands will run as `appuser`.
USER appuser

# The image is now in its "golden" state.
# The default command is to start a shell for interactive training.
CMD ["/bin/bash"]