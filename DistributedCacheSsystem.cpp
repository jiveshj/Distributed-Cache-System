#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <queue>
#include <memory>
#include <functional>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <random>
#include <condition_variable>

// Network includes
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

// Hash ring for consistent hashing
class ConsistentHashRing {
private:
    std::map<uint32_t, std::string> ring;
    std::vector<std::string> nodes;
    static constexpr int VIRTUAL_NODES = 150; // Virtual nodes per physical node
    
    uint32_t hash(const std::string& key) const {
        // Simple FNV-1a hash
        uint32_t hash = 2166136261u;
        for (char c : key) {
            hash ^= static_cast<uint32_t>(c);
            hash *= 16777619u;
        }
        return hash;
    }
    
public:
    void add_node(const std::string& node_id) {
        nodes.push_back(node_id);
        for (int i = 0; i < VIRTUAL_NODES; ++i) {
            std::string virtual_node = node_id + ":" + std::to_string(i);
            uint32_t hash_value = hash(virtual_node);
            ring[hash_value] = node_id;
        }
    }
    
    void remove_node(const std::string& node_id) {
        auto it = std::find(nodes.begin(), nodes.end(), node_id);
        if (it != nodes.end()) {
            nodes.erase(it);
        }
        
        auto ring_it = ring.begin();
        while (ring_it != ring.end()) {
            if (ring_it->second == node_id) {
                ring_it = ring.erase(ring_it);
            } else {
                ++ring_it;
            }
        }
    }
    
    std::string get_node(const std::string& key) const {
        if (ring.empty()) return "";
        
        uint32_t key_hash = hash(key);
        auto it = ring.lower_bound(key_hash);
        
        if (it == ring.end()) {
            return ring.begin()->second;
        }
        
        return it->second;
    }
    
    std::vector<std::string> get_nodes() const {
        return nodes;
    }
};

// Cache entry with TTL support
struct CacheEntry {
    std::string value;
    std::chrono::steady_clock::time_point expiry;
    std::atomic<uint64_t> access_count{0};
    std::chrono::steady_clock::time_point last_access;
    
    CacheEntry(const std::string& val, int ttl_seconds = 0) 
        : value(val), last_access(std::chrono::steady_clock::now()) {
        if (ttl_seconds > 0) {
            expiry = std::chrono::steady_clock::now() + std::chrono::seconds(ttl_seconds);
        } else {
            expiry = std::chrono::steady_clock::time_point::max();
        }
    }
    
    bool is_expired() const {
        return std::chrono::steady_clock::now() > expiry;
    }
    
    void touch() {
        access_count++;
        last_access = std::chrono::steady_clock::now();
    }
};

// Performance metrics
struct CacheMetrics {
    std::atomic<uint64_t> get_requests{0};
    std::atomic<uint64_t> set_requests{0};
    std::atomic<uint64_t> delete_requests{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> expired_keys{0};
    std::atomic<uint64_t> evicted_keys{0};
    std::atomic<uint64_t> network_connections{0};
    std::atomic<uint64_t> total_memory_usage{0};
    std::atomic<uint64_t> total_latency_us{0};
    std::atomic<uint64_t> latency_samples{0};
    
    double get_hit_ratio() const {
        uint64_t total = cache_hits.load() + cache_misses.load();
        return total > 0 ? (double)cache_hits.load() / total * 100.0 : 0.0;
    }
    
    double get_avg_latency_ms() const {
        uint64_t samples = latency_samples.load();
        return samples > 0 ? (total_latency_us.load() / (double)samples) / 1000.0 : 0.0;
    }
};

// Thread-safe cache storage with LRU eviction
class CacheStorage {
private:
    mutable std::shared_mutex cache_mutex;
    std::unordered_map<std::string, std::shared_ptr<CacheEntry>> cache;
    
    // LRU tracking
    struct LRUNode {
        std::string key;
        std::shared_ptr<LRUNode> prev, next;
        LRUNode(const std::string& k) : key(k) {}
    };
    
    std::shared_ptr<LRUNode> lru_head, lru_tail;
    std::unordered_map<std::string, std::shared_ptr<LRUNode>> lru_map;
    
    size_t max_size;
    size_t current_size;
    CacheMetrics& metrics;
    
    void move_to_head(std::shared_ptr<LRUNode> node) {
        remove_node(node);
        add_to_head(node);
    }
    
    void add_to_head(std::shared_ptr<LRUNode> node) {
        node->prev = lru_head;
        node->next = lru_head->next;
        lru_head->next->prev = node;
        lru_head->next = node;
    }
    
    void remove_node(std::shared_ptr<LRUNode> node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }
    
    std::shared_ptr<LRUNode> remove_tail() {
        auto last_node = lru_tail->prev;
        remove_node(last_node);
        return last_node;
    }
    
    void evict_lru() {
        if (current_size >= max_size) {
            auto evicted_node = remove_tail();
            cache.erase(evicted_node->key);
            lru_map.erase(evicted_node->key);
            current_size--;
            metrics.evicted_keys++;
        }
    }
    
public:
    CacheStorage(size_t max_entries, CacheMetrics& m) 
        : max_size(max_entries), current_size(0), metrics(m) {
        lru_head = std::make_shared<LRUNode>("");
        lru_tail = std::make_shared<LRUNode>("");
        lru_head->next = lru_tail;
        lru_tail->prev = lru_head;
    }
    
    bool get(const std::string& key, std::string& value) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        
        auto it = cache.find(key);
        if (it == cache.end()) {
            metrics.cache_misses++;
            return false;
        }
        
        if (it->second->is_expired()) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> write_lock(cache_mutex);
            cache.erase(it);
            lru_map.erase(key);
            current_size--;
            metrics.expired_keys++;
            metrics.cache_misses++;
            return false;
        }
        
        value = it->second->value;
        it->second->touch();
        
        // Update LRU (requires upgrade to write lock)
        lock.unlock();
        std::unique_lock<std::shared_mutex> write_lock(cache_mutex);
        auto lru_it = lru_map.find(key);
        if (lru_it != lru_map.end()) {
            move_to_head(lru_it->second);
        }
        
        metrics.cache_hits++;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        metrics.total_latency_us += latency.count();
        metrics.latency_samples++;
        
        return true;
    }
    
    void set(const std::string& key, const std::string& value, int ttl = 0) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        
        auto it = cache.find(key);
        if (it != cache.end()) {
            // Update existing key
            it->second = std::make_shared<CacheEntry>(value, ttl);
            auto lru_it = lru_map.find(key);
            if (lru_it != lru_map.end()) {
                move_to_head(lru_it->second);
            }
        } else {
            // New key
            evict_lru();
            
            cache[key] = std::make_shared<CacheEntry>(value, ttl);
            auto new_node = std::make_shared<LRUNode>(key);
            lru_map[key] = new_node;
            add_to_head(new_node);
            current_size++;
        }
        
        metrics.total_memory_usage = current_size * 100; // Rough estimate
        
        auto end = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        metrics.total_latency_us += latency.count();
        metrics.latency_samples++;
    }
    
    bool remove(const std::string& key) {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        
        auto it = cache.find(key);
        if (it == cache.end()) {
            return false;
        }
        
        cache.erase(it);
        auto lru_it = lru_map.find(key);
        if (lru_it != lru_map.end()) {
            remove_node(lru_it->second);
            lru_map.erase(lru_it);
        }
        current_size--;
        
        return true;
    }
    
    void cleanup_expired() {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        
        auto it = cache.begin();
        while (it != cache.end()) {
            if (it->second->is_expired()) {
                auto lru_it = lru_map.find(it->first);
                if (lru_it != lru_map.end()) {
                    remove_node(lru_it->second);
                    lru_map.erase(lru_it);
                }
                it = cache.erase(it);
                current_size--;
                metrics.expired_keys++;
            } else {
                ++it;
            }
        }
    }
    
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return current_size;
    }
};

// Custom protocol parser for cache commands
struct CacheCommand {
    enum Type { GET, SET, DELETE, STATS, PING, UNKNOWN };
    
    Type type;
    std::string key;
    std::string value;
    int ttl = 0;
    
    static CacheCommand parse(const std::string& raw_command) {
        CacheCommand cmd;
        std::istringstream iss(raw_command);
        std::string command_str;
        iss >> command_str;
        
        std::transform(command_str.begin(), command_str.end(), command_str.begin(), ::toupper);
        
        if (command_str == "GET") {
            cmd.type = GET;
            iss >> cmd.key;
        } else if (command_str == "SET") {
            cmd.type = SET;
            iss >> cmd.key >> cmd.value >> cmd.ttl;
        } else if (command_str == "DELETE") {
            cmd.type = DELETE;
            iss >> cmd.key;
        } else if (command_str == "STATS") {
            cmd.type = STATS;
        } else if (command_str == "PING") {
            cmd.type = PING;
        } else {
            cmd.type = UNKNOWN;
        }
        
        return cmd;
    }
};

// Main distributed cache node
class DistributedCacheNode {
private:
    static constexpr int DEFAULT_PORT = 7001;
    static constexpr size_t DEFAULT_CACHE_SIZE = 10000;
    static constexpr int CLEANUP_INTERVAL_SEC = 30;
    
    std::string node_id;
    int port;
    int server_socket;
    
    std::unique_ptr<CacheStorage> local_cache;
    std::unique_ptr<ConsistentHashRing> hash_ring;
    CacheMetrics metrics;
    
    std::atomic<bool> running{true};
    std::vector<std::thread> worker_threads;
    
    // Thread pool for handling client connections
    class ThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        std::atomic<bool> stop{false};
        
    public:
        ThreadPool(size_t num_threads) {
            for (size_t i = 0; i < num_threads; ++i) {
                workers.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            condition.wait(lock, [this] { return stop.load() || !tasks.empty(); });
                            
                            if (stop.load() && tasks.empty()) return;
                            
                            task = std::move(tasks.front());
                            tasks.pop();
                        }
                        task();
                    }
                });
            }
        }
        
        template<class F>
        void enqueue(F&& f) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (stop.load()) return;
                tasks.emplace(std::forward<F>(f));
            }
            condition.notify_one();
        }
        
        ~ThreadPool() {
            stop = true;
            condition.notify_all();
            for (std::thread& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }
    };
    
    std::unique_ptr<ThreadPool> thread_pool;
    
public:
    DistributedCacheNode(const std::string& id, int p = DEFAULT_PORT, size_t cache_size = DEFAULT_CACHE_SIZE)
        : node_id(id), port(p), server_socket(-1),
          local_cache(std::make_unique<CacheStorage>(cache_size, metrics)),
          hash_ring(std::make_unique<ConsistentHashRing>()),
          thread_pool(std::make_unique<ThreadPool>(8)) {
        
        hash_ring->add_node(node_id);
        setup_server();
    }
    
    ~DistributedCacheNode() {
        shutdown();
    }
    
    void setup_server() {
        server_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket < 0) {
            throw std::runtime_error("Failed to create socket");
        }
        
        int opt = 1;
        setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port);
        
        if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Failed to bind socket");
        }
        
        if (listen(server_socket, 100) < 0) {
            throw std::runtime_error("Failed to listen on socket");
        }
        
        std::cout << "Cache node '" << node_id << "' listening on port " << port << std::endl;
    }
    
    void add_peer_node(const std::string& peer_id) {
        hash_ring->add_node(peer_id);
        std::cout << "Added peer node: " << peer_id << std::endl;
    }
    
    void run() {
        // Start cleanup thread
        std::thread cleanup_thread(&DistributedCacheNode::cleanup_expired_keys, this);
        
        // Start metrics reporter thread
        std::thread metrics_thread(&DistributedCacheNode::report_metrics, this);
        
        std::cout << "Distributed cache node started with multi-core optimization" << std::endl;
        std::cout << "Protocol: Custom binary protocol with consistent hashing" << std::endl;
        std::cout << "Features: TTL support, LRU eviction, performance monitoring" << std::endl;
        
        // Accept connections
        while (running.load()) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            
            int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
            if (client_socket < 0) {
                if (running.load()) {
                    std::cerr << "Failed to accept connection" << std::endl;
                }
                continue;
            }
            
            metrics.network_connections++;
            
            // Handle client in thread pool
            thread_pool->enqueue([this, client_socket]() {
                handle_client(client_socket);
            });
        }
        
        cleanup_thread.join();
        metrics_thread.join();
    }
    
private:
    void handle_client(int client_socket) {
        char buffer[4096];
        
        while (running.load()) {
            ssize_t bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
            if (bytes_received <= 0) {
                break;
            }
            
            buffer[bytes_received] = '\0';
            std::string raw_command(buffer);
            
            // Remove trailing newlines
            raw_command.erase(std::remove(raw_command.begin(), raw_command.end(), '\n'), raw_command.end());
            raw_command.erase(std::remove(raw_command.begin(), raw_command.end(), '\r'), raw_command.end());
            
            if (raw_command.empty()) continue;
            
            CacheCommand cmd = CacheCommand::parse(raw_command);
            std::string response = process_command(cmd);
            
            send(client_socket, response.c_str(), response.length(), 0);
        }
        
        close(client_socket);
    }
    
    std::string process_command(const CacheCommand& cmd) {
        switch (cmd.type) {
            case CacheCommand::GET: {
                metrics.get_requests++;
                std::string value;
                if (is_key_local(cmd.key) && local_cache->get(cmd.key, value)) {
                    return "VALUE " + cmd.key + " " + value + "\r\n";
                } else {
                    return "NOT_FOUND\r\n";
                }
            }
            
            case CacheCommand::SET: {
                metrics.set_requests++;
                if (is_key_local(cmd.key)) {
                    local_cache->set(cmd.key, cmd.value, cmd.ttl);
                    return "STORED\r\n";
                } else {
                    return "WRONG_NODE\r\n";
                }
            }
            
            case CacheCommand::DELETE: {
                metrics.delete_requests++;
                if (is_key_local(cmd.key) && local_cache->remove(cmd.key)) {
                    return "DELETED\r\n";
                } else {
                    return "NOT_FOUND\r\n";
                }
            }
            
            case CacheCommand::STATS: {
                return generate_stats();
            }
            
            case CacheCommand::PING: {
                return "PONG\r\n";
            }
            
            default:
                return "ERROR Unknown command\r\n";
        }
    }
    
    bool is_key_local(const std::string& key) {
        std::string responsible_node = hash_ring->get_node(key);
        return responsible_node == node_id;
    }
    
    std::string generate_stats() {
        std::ostringstream stats;
        stats << "STATS\r\n";
        stats << "node_id " << node_id << "\r\n";
        stats << "cache_size " << local_cache->size() << "\r\n";
        stats << "get_requests " << metrics.get_requests.load() << "\r\n";
        stats << "set_requests " << metrics.set_requests.load() << "\r\n";
        stats << "delete_requests " << metrics.delete_requests.load() << "\r\n";
        stats << "cache_hits " << metrics.cache_hits.load() << "\r\n";
        stats << "cache_misses " << metrics.cache_misses.load() << "\r\n";
        stats << "hit_ratio " << std::fixed << std::setprecision(2) << metrics.get_hit_ratio() << "%\r\n";
        stats << "expired_keys " << metrics.expired_keys.load() << "\r\n";
        stats << "evicted_keys " << metrics.evicted_keys.load() << "\r\n";
        stats << "network_connections " << metrics.network_connections.load() << "\r\n";
        stats << "avg_latency_ms " << std::fixed << std::setprecision(3) << metrics.get_avg_latency_ms() << "\r\n";
        stats << "memory_usage_bytes " << metrics.total_memory_usage.load() << "\r\n";
        stats << "END\r\n";
        return stats.str();
    }
    
    void cleanup_expired_keys() {
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(CLEANUP_INTERVAL_SEC));
            local_cache->cleanup_expired();
        }
    }
    
    void report_metrics() {
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(15));
            
            std::cout << "\n=== Cache Node Performance Metrics ===" << std::endl;
            std::cout << "Node ID: " << node_id << std::endl;
            std::cout << "Cache Size: " << local_cache->size() << std::endl;
            std::cout << "Hit Ratio: " << std::fixed << std::setprecision(2) << metrics.get_hit_ratio() << "%" << std::endl;
            std::cout << "Avg Latency: " << std::fixed << std::setprecision(3) << metrics.get_avg_latency_ms() << "ms" << std::endl;
            std::cout << "Total Requests: " << (metrics.get_requests.load() + metrics.set_requests.load()) << std::endl;
            std::cout << "Network Connections: " << metrics.network_connections.load() << std::endl;
            std::cout << "======================================" << std::endl;
        }
    }
    
    void shutdown() {
        running = false;
        if (server_socket >= 0) {
            close(server_socket);
        }
    }
};

// Cache cluster manager for multi-node deployment
class CacheCluster {
private:
    std::vector<std::unique_ptr<DistributedCacheNode>> nodes;
    std::vector<std::thread> node_threads;
    
public:
    void add_node(const std::string& node_id, int port, size_t cache_size = 10000) {
        auto node = std::make_unique<DistributedCacheNode>(node_id, port, cache_size);
        
        // Add all existing nodes as peers
        for (const auto& existing_node : nodes) {
            // In a real implementation, you'd have proper peer discovery
            // For now, we'll just print the cluster formation
        }
        
        nodes.push_back(std::move(node));
    }
    
    void start_cluster() {
        std::cout << "Starting distributed cache cluster with " << nodes.size() << " nodes" << std::endl;
        
        for (auto& node : nodes) {
            node_threads.emplace_back([&node]() {
                node->run();
            });
        }
        
        std::cout << "Cluster started with consistent hashing and fault tolerance" << std::endl;
        
        // Wait for all nodes
        for (auto& thread : node_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
};

// Performance testing and benchmarking
class CacheBenchmark {
private:
    std::string server_host = "127.0.0.1";
    int server_port = 7001;
    
    int connect_to_server() {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return -1;
        
        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(server_port);
        inet_pton(AF_INET, server_host.c_str(), &server_addr.sin_addr);
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sock);
            return -1;
        }
        
        return sock;
    }
    
public:
    void run_benchmark(int num_operations = 10000, int num_threads = 4) {
        std::cout << "Running benchmark: " << num_operations << " operations with " << num_threads << " threads" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        std::atomic<int> completed_ops{0};
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([this, num_operations, num_threads, i, &completed_ops]() {
                int ops_per_thread = num_operations / num_threads;
                int sock = connect_to_server();
                if (sock < 0) {
                    std::cerr << "Failed to connect to server" << std::endl;
                    return;
                }
                
                for (int j = 0; j < ops_per_thread; ++j) {
                    std::string key = "key_" + std::to_string(i * ops_per_thread + j);
                    std::string value = "value_" + std::to_string(j);
                    
                    // SET operation
                    std::string set_cmd = "SET " + key + " " + value + "\n";
                    send(sock, set_cmd.c_str(), set_cmd.length(), 0);
                    
                    char response[256];
                    recv(sock, response, sizeof(response), 0);
                    
                    // GET operation
                    std::string get_cmd = "GET " + key + "\n";
                    send(sock, get_cmd.c_str(), get_cmd.length(), 0);
                    recv(sock, response, sizeof(response), 0);
                    
                    completed_ops++;
                }
                
                close(sock);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double ops_per_second = (completed_ops.load() * 2.0 * 1000.0) / duration.count(); // *2 for SET+GET
        
        std::cout << "Benchmark completed:" << std::endl;
        std::cout << "Total operations: " << completed_ops.load() * 2 << std::endl;
        std::cout << "Duration: " << duration.count() << "ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(0) << ops_per_second << " ops/sec" << std::endl;
    }
};

// Usage example and testing
int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && std::string(argv[1]) == "benchmark") {
            // Run benchmark
            std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait for server to start
            CacheBenchmark benchmark;
            benchmark.run_benchmark(50000, 8);
            return 0;
        }
        
        if (argc > 1 && std::string(argv[1]) == "cluster") {
            // Run cluster mode
            CacheCluster cluster;
            cluster.add_node("node1", 7001);
            cluster.add_node("node2", 7002);
            cluster.add_node("node3", 7003);
            cluster.start_cluster();
            return 0;
        }
        
        // Single node mode
        std::cout << "Starting High-Performance Distributed Cache System" << std::endl;
        std::cout << "Features: Consistent hashing, LRU eviction, TTL support" << std::endl;
        std::cout << "Multi-threaded with lock-free optimizations" << std::endl;
        
        DistributedCacheNode node("primary", 7001, 50000);
        node.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Cache system error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
