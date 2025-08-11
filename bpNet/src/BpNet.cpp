#include "bpNet.h"
#include <cmath>
#include <cstring>  // For memcpy
#include <algorithm>

// --- 构造函数 ---
bpNet::bpNet(int i_node, int h_node, int o_node, double learn_rate) :
    i_node(i_node),
    h_node(h_node),
    o_node(o_node),
    learn_rate(learn_rate),
    gen(std::random_device{}()),
    dis(-0.5, 0.5)
{
    // 计算所需总内存
    size_t total_size = 
        i_node * h_node +  // ih_w
        h_node * o_node +  // ho_w
        h_node +           // h_b
        o_node +           // o_b
        i_node +           // i_data
        h_node +           // h_out
        o_node +           // o_out
        h_node +           // h_grad
        o_node;            // o_grad
    
    // 分配连续内存
    all_memory.resize(total_size, 0.0);
    double* ptr = all_memory.data();
    
    // 设置指针
    ih_w = ptr; ptr += i_node * h_node;
    ho_w = ptr; ptr += h_node * o_node;
    h_b = ptr; ptr += h_node;
    o_b = ptr; ptr += o_node;
    i_data = ptr; ptr += i_node;
    h_out = ptr; ptr += h_node;
    o_out = ptr; ptr += o_node;
    h_grad = ptr; ptr += h_node;
    o_grad = ptr;
    
    // 初始化权重
    init();
}

// --- 析构函数 ---
bpNet::~bpNet() {
    // 不需要手动释放内存，vector会自动管理
}

// --- 初始化权重 ---
void bpNet::init() {
    // 初始化权重和偏置
    for (int i = 0; i < i_node * h_node; ++i) ih_w[i] = dis(gen);
    for (int i = 0; i < h_node; ++i) h_b[i] = dis(gen);
    for (int i = 0; i < h_node * o_node; ++i) ho_w[i] = dis(gen);
    for (int i = 0; i < o_node; ++i) o_b[i] = dis(gen);
}

// --- 快速Sigmoid函数 ---
inline double bpNet::sigmoid(double z) const {
    // 精确计算版本（避免近似带来的精度问题）
    return 1.0 / (1.0 + std::exp(-z));
}

// --- 前向传播 ---
void bpNet::forward_propa(const std::vector<double>& input_data) {
    // 复制输入数据
    std::memcpy(i_data, input_data.data(), i_node * sizeof(double));
    
    // 计算隐藏层输出
    for (int h = 0; h < h_node; ++h) {
        double sum = 0.0;
        const int base_idx = h * i_node;
        
        // 手动循环展开
        int i = 0;
        for (; i <= i_node - 4; i += 4) {
            sum += i_data[i] * ih_w[base_idx + i];
            sum += i_data[i+1] * ih_w[base_idx + i+1];
            sum += i_data[i+2] * ih_w[base_idx + i+2];
            sum += i_data[i+3] * ih_w[base_idx + i+3];
        }
        
        // 处理剩余元素
        for (; i < i_node; ++i) {
            sum += i_data[i] * ih_w[base_idx + i];
        }
        
        h_out[h] = sigmoid(sum + h_b[h]);
    }
    
    // 计算输出层输出
    for (int o = 0; o < o_node; ++o) {
        double sum = 0.0;
        const int base_idx = o * h_node;
        
        // 手动循环展开
        int h = 0;
        for (; h <= h_node - 4; h += 4) {
            sum += h_out[h] * ho_w[base_idx + h];
            sum += h_out[h+1] * ho_w[base_idx + h+1];
            sum += h_out[h+2] * ho_w[base_idx + h+2];
            sum += h_out[h+3] * ho_w[base_idx + h+3];
        }
        
        // 处理剩余元素
        for (; h < h_node; ++h) {
            sum += h_out[h] * ho_w[base_idx + h];
        }
        
        o_out[o] = sigmoid(sum + o_b[o]);
    }
}

// --- 反向传播 ---
void bpNet::back_propa(const std::vector<double>& target_data) {
    // 计算输出层梯度
    for (int o = 0; o < o_node; ++o) {
        const double error = target_data[o] - o_out[o];
        const double sig_deriv = o_out[o] * (1.0 - o_out[o]);
        o_grad[o] = error * sig_deriv;
    }
    
    // 计算隐藏层梯度
    for (int h = 0; h < h_node; ++h) {
        double sum = 0.0;
        for (int o = 0; o < o_node; ++o) {
            sum += o_grad[o] * ho_w[o * h_node + h];
        }
        h_grad[h] = sum * h_out[h] * (1.0 - h_out[h]);
    }
    
    // 更新隐藏层到输出层权重
    for (int o = 0; o < o_node; ++o) {
        const double grad = learn_rate * o_grad[o];
        const int base_idx = o * h_node;
        
        for (int h = 0; h < h_node; ++h) {
            ho_w[base_idx + h] += grad * h_out[h];
        }
        o_b[o] += grad;
    }
    
    // 更新输入层到隐藏层权重
    for (int h = 0; h < h_node; ++h) {
        const double grad = learn_rate * h_grad[h];
        const int base_idx = h * i_node;
        
        for (int i = 0; i < i_node; ++i) {
            ih_w[base_idx + i] += grad * i_data[i];
        }
        h_b[h] += grad;
    }
}

// --- 训练接口 ---
void bpNet::train(const std::vector<double>& i_data, const std::vector<double>& t_data) {
    forward_propa(i_data);
    back_propa(t_data);
}

// --- 预测接口 ---
std::vector<double> bpNet::predict(const std::vector<double>& i_data) {
    forward_propa(i_data);
    return std::vector<double>(o_out, o_out + o_node);
}

// --- 批量预测 ---
std::vector<std::vector<double>> bpNet::predict_batch(const std::vector<std::vector<double>>& inputs) {
    std::vector<std::vector<double>> results;
    results.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        forward_propa(input);
        results.emplace_back(o_out, o_out + o_node);
    }
    
    return results;
}

// --- 计算损失 ---
double bpNet::get_loss(const std::vector<double>& t_data) const {
    double loss = 0.0;
    for (int o = 0; o < o_node; ++o) {
        double diff = t_data[o] - o_out[o];
        loss += diff * diff;
    }
    return loss * 0.5;
}