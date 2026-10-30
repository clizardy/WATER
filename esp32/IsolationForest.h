#ifndef ISOLATION_FOREST_H
#define ISOLATION_FOREST_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>

struct Node {
    int feature = -1;       // -1 means leaf node
    float split_val = 0.0f;
    Node* left = nullptr;
    Node* right = nullptr;
    int size = 0;           // Number of samples reaching this node

    ~Node() {
        delete left;
        delete right;
    }
};

class IsolationForest {
private:
    int num_trees;
    int max_samples;
    int max_height;
    std::vector<Node*> forest;
    std::mt19937 rng;

    // Average path length of unsuccessful search in a Binary Search Tree (BST)
    float c(int n) {
        if (n <= 1) return 0.0f;
        if (n == 2) return 1.0f;
        return 2.0f * (log(n - 1) + 0.5772156649f) - (2.0f * (n - 1) / n);
    }

    Node* buildTree(const std::vector<std::vector<float>>& data, const std::vector<int>& indices, int height, int limit) {
        Node* node = new Node();
        node->size = indices.size();

        // Termination conditions: max height reached or node has <= 1 sample
        if (height >= limit || indices.size() <= 1) {
            return node;
        }

        int dims = data[0].size();
        
        // 1. Pick a random feature dimension
        std::uniform_int_distribution<int> dist_dim(0, dims - 1);
        int f = dist_dim(rng);

        // 2. Find min and max values for feature f among current samples
        float min_val = data[indices[0]][f];
        float max_val = min_val;
        for (size_t i = 1; i < indices.size(); ++i) {
            float val = data[indices[i]][f];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        // If all values along this feature are identical, it's a leaf
        if (min_val == max_val) {
            return node;
        }

        // 3. Pick a random split value between min and max
        std::uniform_real_distribution<float> dist_val(min_val, max_val);
        float p = dist_val(rng);

        node->feature = f;
        node->split_val = p;

        // 4. Partition data into left (< p) and right (>= p)
        std::vector<int> left_indices;
        std::vector<int> right_indices;
        for (int idx : indices) {
            if (data[idx][f] < p) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        // Edge case: if all samples fall on one side
        if (left_indices.empty() || right_indices.empty()) {
            node->feature = -1;
            return node;
        }

        node->left = buildTree(data, left_indices, height + 1, limit);
        node->right = buildTree(data, right_indices, height + 1, limit);
        return node;
    }

    float getPathLength(const std::vector<float>& x, Node* node, int height) {
        // If leaf node or split feature out of bounds
        if (node->feature == -1 || node->left == nullptr || node->right == nullptr) {
            return height + c(node->size);
        }
        
        int f = node->feature;
        if (x[f] < node->split_val) {
            return getPathLength(x, node->left, height + 1);
        } else {
            return getPathLength(x, node->right, height + 1);
        }
    }

    // Recursively save a node to binary stream
    void saveNode(const Node* node, std::ostream& out) const {
        if (!node) {
            bool has_node = false;
            out.write(reinterpret_cast<const char*>(&has_node), sizeof(has_node));
            return;
        }
        bool has_node = true;
        out.write(reinterpret_cast<const char*>(&has_node), sizeof(has_node));
        out.write(reinterpret_cast<const char*>(&node->feature), sizeof(node->feature));
        out.write(reinterpret_cast<const char*>(&node->split_val), sizeof(node->split_val));
        out.write(reinterpret_cast<const char*>(&node->size), sizeof(node->size));
        
        saveNode(node->left, out);
        saveNode(node->right, out);
    }

    // Recursively load a node from binary stream
    Node* loadNode(std::istream& in) {
        bool has_node = false;
        if (!in.read(reinterpret_cast<char*>(&has_node), sizeof(has_node)) || !has_node) {
            return nullptr;
        }
        Node* node = new Node();
        in.read(reinterpret_cast<char*>(&node->feature), sizeof(node->feature));
        in.read(reinterpret_cast<char*>(&node->split_val), sizeof(node->split_val));
        in.read(reinterpret_cast<char*>(&node->size), sizeof(node->size));

        node->left = loadNode(in);
        node->right = loadNode(in);
        return node;
    }

public:
    IsolationForest(int n_trees = 10, int n_samples = 128) 
        : num_trees(n_trees), max_samples(n_samples) {
        max_height = ceil(log2(std::max(2, max_samples)));
        
        // Seed RNG using ESP32 hardware random or fallback
        #ifdef ESP32
        rng.seed(esp_random());
        #else
        rng.seed(std::random_device()());
        #endif
    }

    ~IsolationForest() {
        reset();
    }

    void reset() {
        for (Node* tree : forest) {
            delete tree;
        }
        forest.clear();
    }

    // --- TRAINING ---
    void train(const std::vector<std::vector<float>>& dataset) {
        reset();
        if (dataset.empty()) return;

        int n_total = dataset.size();
        int actual_samples = std::min(n_total, max_samples);

        for (int i = 0; i < num_trees; ++i) {
            // Subsample dataset indices randomly
            std::vector<int> indices(n_total);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            indices.resize(actual_samples);

            Node* root = buildTree(dataset, indices, 0, max_height);
            forest.push_back(root);
        }
    }

    // --- INFERENCE / SCORING ---
    // Returns an anomaly score between 0.0 and 1.0.
    // Scores closer to 1.0 indicate strong anomalies. Scores < 0.5 indicate normal data.
    float score(const std::vector<float>& sample) {
        if (forest.empty()) return 0.0f;
        
        float sum_path = 0.0f;
        int n = max_samples;
        float denominator = c(n);
        if (denominator == 0.0f) denominator = 1.0f;

        for (Node* tree : forest) {
            sum_path += getPathLength(sample, tree, 0);
        }

        float avg_path = sum_path / forest.size();
        return pow(2.0f, -(avg_path / denominator));
    }

    // Save trained model to a binary file
    bool save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) return false;

        // Save metadata
        out.write(reinterpret_cast<const char*>(&num_trees), sizeof(num_trees));
        out.write(reinterpret_cast<const char*>(&max_samples), sizeof(max_samples));
        out.write(reinterpret_cast<const char*>(&max_height), sizeof(max_height));

        // Save each tree in the forest
        for (const Node* tree : forest) {
            saveNode(tree, out);
        }
        return true;
    }

    // Load trained model from a binary file
    bool load(const std::string& filename) {
        reset(); // Clear existing forest
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;

        // Load metadata
        in.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
        in.read(reinterpret_cast<char*>(&max_samples), sizeof(max_samples));
        in.read(reinterpret_cast<char*>(&max_height), sizeof(max_height));

        // Rebuild each tree in the forest
        for (int i = 0; i < num_trees; ++i) {
            Node* root = loadNode(in);
            forest.push_back(root);
        }
        return true;
    }
};

#endif

