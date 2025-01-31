# Use the NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies (Rust, essential build tools)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set the Rust environment
ENV PATH=/root/.cargo/bin:$PATH

# Clone the repository
RUN git clone https://github.com/gouthamsk98/speech-llm.git

# Navigate to the 'rust' directory and build the project
WORKDIR /speech-llm/rust
RUN cargo build --release

# Expose port 80
EXPOSE 80

# Retrieve the API key from environment variable
ENV API_KEY=${API_KEY}

# Run the built project with API key from environment variable
CMD ["./target/release/rust", "--serve", "--host", "0.0.0.0", "--port", "80", "api-key", "${API_KEY}"]
