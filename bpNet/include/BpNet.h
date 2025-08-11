#ifndef BPNET_H
#define BPNET_H

#include <vector>
#include <random>

class bpNet {
public:
    bpNet(int i_node, int h_node, int o_node, double learn_rate);
    ~bpNet();
    
    void train(const std::vector<double>& i_data, const std::vector<double>& t_data);
    std::vector<double> predict(const std::vector<double>& i_data);
    double get_loss(const std::vector<double>& t_data) const;
    std::vector<std::vector<double>> predict_batch(const std::vector<std::vector<double>>& inputs);

private:
    void init();
    void forward_propa(const std::vector<double>& input_data);
    void back_propa(const std::vector<double>& target_data);
    inline double sigmoid(double z) const;

    // 网络结构参数
    int i_node;         // 输入层节点数
    int h_node;         // 隐藏层节点数
    int o_node;         // 输出层节点数
    double learn_rate;  // 学习率

    // 权重和偏置（一维数组存储）
    double* ih_w;       // 输入层到隐藏层权重 [i_node * h_node]
    double* ho_w;       // 隐藏层到输出层权重 [h_node * o_node]
    double* h_b;        // 隐藏层偏置 [h_node]
    double* o_b;        // 输出层偏置 [o_node]
    
    // 网络状态
    double* i_data;     // 输入数据 [i_node]
    double* h_out;      // 隐藏层输出 [h_node]
    double* o_out;      // 输出层输出 [o_node]
    
    // 梯度
    double* h_grad;     // 隐藏层梯度 [h_node]
    double* o_grad;     // 输出层梯度 [o_node]
    
    // 内存管理
    std::vector<double> all_memory; // 所有内存的统一容器
    
    // 随机数生成
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
};

#endif // BPNET_H