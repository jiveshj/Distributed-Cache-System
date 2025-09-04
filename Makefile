# Distributed Cache System Makefile
# Optimized for multi-core compilation and performance

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mtune=native
LDFLAGS = -pthread
TARGET = distributed_cache
SOURCES = cache_system.cpp

# Performance optimization flags
PERF_FLAGS = -DNDEBUG -flto -ffast-math -funroll-loops
DEBUG_FLAGS = -g -O0 -DDEBUG -fsanitize=address -fsanitize=thread

# Default target
all: release

# Release build with maximum optimization
release: CXXFLAGS += $(PERF_FLAGS)
release: $(TARGET)

# Debug build with sanitizers
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: LDFLAGS += -fsanitize=address -fsanitize=thread
debug: $(TARGET)

# Performance profiling build
profile: CXXFLAGS += -pg -O2
profile: LDFLAGS += -pg
profile: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Benchmarking targets
benchmark: release
	@echo "Building benchmark suite..."
	@./$(TARGET) benchmark

# Load testing
load-test: release
	@echo "Running load test with 1000 concurrent connections..."
	@for i in {1..1000}; do \
		(echo "SET key$$i value$$i"; echo "GET key$$i") | nc localhost 7001 & \
	done
	@wait
	@echo "Load test completed"

# Performance profiling
perf-analysis: profile
	@echo "Running performance analysis..."
	@perf record -g ./$(TARGET) &
	@sleep 30
	@pkill -f $(TARGET)
	@perf report

# Memory leak detection
valgrind: debug
	valgrind --tool=memcheck --leak-check=full --track-origins=yes ./$(TARGET)

# Static analysis
static-analysis:
	@echo "Running static analysis..."
	@cppcheck --enable=all --std=c++17 $(SOURCES)
	@clang-static-analyzer $(SOURCES)

# Docker targets
docker-build:
	docker build -t distributed-cache:latest .

docker-run: docker-build
	docker run -p 7001:7001 distributed-cache:latest

# Kubernetes targets
k8s-deploy:
	kubectl apply -f k8s-config.yaml

k8s-scale:
	kubectl scale statefulset distributed-cache --replicas=5

k8s-status:
	kubectl get pods -l app=distributed-cache
	kubectl get services cache-service

# Cluster deployment
cluster-deploy:
	@echo "Deploying 3-node cluster..."
	@./$(TARGET) cluster &
	@echo "Cluster deployed. Access nodes at ports 7001, 7002, 7003"

# Performance monitoring
monitor:
	@echo "Cache Performance Monitor"
	@echo "========================"
	@while true; do \
		echo "STATS" | nc localhost 7001; \
		sleep 5; \
		clear; \
	done

# Cleanup
clean:
	rm -f $(TARGET)
	rm -f gmon.out
	rm -f perf.data*
	docker rmi distributed-cache:latest 2>/dev/null || true

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y build-essential cmake netcat-openbsd valgrind cppcheck
	sudo apt-get install -y docker.io kubectl

# System tuning for performance
tune-system:
	@echo "Applying system tuning for high performance..."
	@echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf
	@echo 'net.ipv4.tcp_max_syn_backlog = 65535' | sudo tee -a /etc/sysctl.conf
	@echo 'fs.file-max = 1000000' | sudo tee -a /etc/sysctl.conf
	@sudo sysctl -p
	@echo "System tuned for high-performance networking"

# Continuous Integration
ci: static-analysis release benchmark
	@echo "All CI checks passed!"

.PHONY: all release debug profile benchmark load-test perf-analysis valgrind static-analysis
.PHONY: docker-build docker-run k8s-deploy k8s-scale k8s-status cluster-deploy
.PHONY: monitor clean install-deps tune-system ci
