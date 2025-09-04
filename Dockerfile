# Multi-stage build for optimized production image
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY cache_system.cpp .
COPY CMakeLists.txt .

# Build the application with optimizations
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Production stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r cache && useradd -r -g cache cache

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/build/distributed_cache /app/
COPY --chown=cache:cache --from=builder /app/build/distributed_cache /app/

# Create data directory
RUN mkdir -p /app/data && chown cache:cache /app/data

# Switch to non-root user
USER cache

# Expose cache port
EXPOSE 7001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD echo "PING" | nc localhost 7001 | grep -q "PONG" || exit 1

# Set resource limits and performance tuning
ENV CACHE_MAX_MEMORY=1GB
ENV CACHE_MAX_CONNECTIONS=1000
ENV CACHE_THREAD_POOL_SIZE=8

# Start the cache server
CMD ["./distributed_cache"]
