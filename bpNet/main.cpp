// MNIST数据集测试
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>    // 用于高精度计时
#include <limits>    // 用于清空输入缓冲区
#include <Windows.h> // 用于设置控制台编码
#include <random>    // 用于随机采样
#include <algorithm>
#include "bpNet.h"
#include "MNISTLoader.h"

// 随机采样函数
std::vector<std::vector<double>> random_sample(
    const std::vector<std::vector<double>>& inputs, 
    const std::vector<std::vector<double>>& targets,
    size_t sample_size,
    std::vector<std::vector<double>>& sampled_targets)
{
    // 确保输入和目标大小一致
    if (inputs.size() != targets.size()) {
        throw std::runtime_error("输入和目标大小不一致");
    }
    
    // 如果请求的样本大小大于数据集大小，则使用整个数据集
    if (sample_size >= inputs.size()) {
        sampled_targets = targets;
        return inputs;
    }
    
    // 创建索引向量
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // 随机打乱索引
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // 创建采样结果
    std::vector<std::vector<double>> sampled_inputs;
    sampled_targets.clear();
    
    sampled_inputs.reserve(sample_size);
    sampled_targets.reserve(sample_size);
    
    for (size_t i = 0; i < sample_size; i++) {
        sampled_inputs.push_back(inputs[indices[i]]);
        sampled_targets.push_back(targets[indices[i]]);
    }
    
    return sampled_inputs;
}

int main() {
    // 设置控制台输出编码为UTF-8
    SetConsoleOutputCP(65001);

    // --- 1. 定义网络结构和数据文件 ---
    const int input_nodes = 784;   // MINST 28x28
    const int hidden_nodes = 64;
    const int output_nodes = 10;
    const double learning_rate = 0.1;
    const size_t MAX_TRAIN_SAMPLES = 3000; // 最大训练样本数
    const size_t MAX_TEST_SAMPLES = 1000;  // 最大测试样本数
    
    // MINIST 数据集文件路径
    std::string train_images_file = "data/train-images.idx3-ubyte";
    std::string train_labels_file = "data/train-labels.idx1-ubyte";
    std::string test_images_file = "data/t10k-images.idx3-ubyte";
    std::string test_labels_file = "data/t10k-labels.idx1-ubyte";

    // --- 2. 加载完整数据集 ---
    std::vector<std::vector<double>> full_train_inputs;
    std::vector<std::vector<double>> full_train_targets;
    std::vector<std::vector<double>> full_test_inputs;
    std::vector<std::vector<double>> full_test_targets;
    
    std::cout << "--- 数据加载阶段 ---" << std::endl;
    
    // 加载完整训练数据
    std::cout << "正在加载完整 MNIST 训练数据集..." << std::endl;
    if (!MNISTLoader::load_images(train_images_file, full_train_inputs, 0)) {
        std::cerr << "训练数据加载失败，程序即将退出。" << std::endl;
        std::cin.get();
        return 1;
    }
    if (!MNISTLoader::load_labels(train_labels_file, full_train_targets, 0)) {
        std::cerr << "训练标签加载失败，程序即将退出。" << std::endl;
        std::cin.get();
        return 1;
    }
    std::cout << "完整训练数据集加载成功！共 " << full_train_inputs.size() << " 条样本。" << std::endl;
    
    // 加载完整测试数据
    std::cout << "正在加载完整 MNIST 测试数据集..." << std::endl;
    if (!MNISTLoader::load_images(test_images_file, full_test_inputs, 0)) {
        std::cerr << "测试数据加载失败，程序即将退出。" << std::endl;
        std::cin.get();
        return 1;
    }
    if (!MNISTLoader::load_labels(test_labels_file, full_test_targets, 0)) {
        std::cerr << "测试标签加载失败，程序即将退出。" << std::endl;
        std::cin.get();
        return 1;
    }
    std::cout << "完整测试数据集加载成功！共 " << full_test_inputs.size() << " 条样本。" << std::endl;

    // 从完整数据集中随机采样
    std::vector<std::vector<double>> train_inputs;
    std::vector<std::vector<double>> train_targets;
    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_targets;
    
    std::cout << "\n正在从训练数据集中随机采样 " << MAX_TRAIN_SAMPLES << " 条样本..." << std::endl;
    train_inputs = random_sample(full_train_inputs, full_train_targets, MAX_TRAIN_SAMPLES, train_targets);
    
    std::cout << "正在从测试数据集中随机采样 " << MAX_TEST_SAMPLES << " 条样本..." << std::endl;
    test_inputs = random_sample(full_test_inputs, full_test_targets, MAX_TEST_SAMPLES, test_targets);
    
    // 检查采样结果
    if (train_inputs.empty() || test_inputs.empty()) {
        std::cerr << "随机采样失败，程序即将退出。" << std::endl;
        std::cin.get();
        return 1;
    }
    
    // 输出样本统计信息
    std::cout << "\n数据集统计信息:" << std::endl;
    std::cout << " - 训练样本数: " << train_inputs.size() << std::endl;
    std::cout << " - 测试样本数: " << test_inputs.size() << std::endl;
    
    // --- 3. 获取用户输入参数 ---
    double target_error;
    int max_epochs;
    std::cout << "\n--- 训练参数设置 ---" << std::endl;
    std::cout << "请输入您的目标平均损失 (例如 0.001): ";
    while (!(std::cin >> target_error) || target_error <= 0) {
        std::cout << "输入无效，请输入一个大于0的数字: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::cout << "请输入最大训练轮数 (例如 50000): ";
    while (!(std::cin >> max_epochs) || max_epochs <= 0) {
        std::cout << "输入无效，请输入一个正整数: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // --- 4. 创建网络并开始训练 ---
    bpNet network(input_nodes, hidden_nodes, output_nodes, learning_rate);
    std::cout << "\n--- 网络训练阶段 ---" << std::endl;
    std::cout << "参数已设定。按回车键开始训练..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    auto start_time = std::chrono::high_resolution_clock::now(); // 开始计时

    bool target_achieved = false;
    int final_epoch = 0;
    double current_loss = 1.0;

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        double epoch_loss = 0;
        
        // 训练过程
        for (size_t i = 0; i < train_inputs.size(); ++i) {
            network.train(train_inputs[i], train_targets[i]);
            epoch_loss += network.get_loss(train_targets[i]);
        }
        
        current_loss = epoch_loss / train_inputs.size();
        final_epoch = epoch;

        // 简洁的进度显示 - 只显示关键信息
        if (epoch % 10 == 0 || epoch == 1 || epoch == max_epochs) {
            std::cout << "\r训练进度: 轮次 " << std::setw(6) << epoch << "/" << max_epochs
                      << " | 平均损失: " << std::fixed << std::setprecision(8) << current_loss
                      << std::flush;
        }

        // 检查是否达到目标误差
        if (current_loss <= target_error) {
            target_achieved = true;
            break; // 达到目标，跳出循环
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now(); // 结束计时
    std::chrono::duration<double> training_duration = end_time - start_time;

    std::cout << std::endl; // 训练结束后换行，保留最后一条进度信息

    // --- 5. 输出最终训练结果 ---
    std::cout << "\n--- 训练结果总结 ---" << std::endl;
    if (target_achieved) {
        std::cout << "训练成功！已达到目标误差。" << std::endl;
    } else {
        std::cout << "训练停止。已达到最大训练轮数但未达到目标误差。" << std::endl;
    }
    std::cout << " - 目标误差: " << target_error << std::endl;
    std::cout << " - 最终误差: " << current_loss << std::endl;
    std::cout << " - 所用轮次: " << final_epoch << " / " << max_epochs << std::endl;
    std::cout << " - 训练耗时: " << std::fixed << std::setprecision(3) << training_duration.count() << " 秒" << std::endl;

    // --- 6. 在测试集上进行评估 ---
    std::cout << "\n正在测试模型性能..." << std::endl;
    
    int total_correct = 0;
    std::vector<int> class_correct(10, 0); // 每个类别的正确计数
    std::vector<int> class_total(10, 0);   // 每个类别的总样本数
    
    // 对测试集进行预测
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        std::vector<double> prediction = network.predict(test_inputs[i]);
        int predicted_label = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        int actual_label = std::distance(test_targets[i].begin(), std::max_element(test_targets[i].begin(), test_targets[i].end()));
        
        // 更新类别计数
        class_total[actual_label]++;
        if (predicted_label == actual_label) {
            total_correct++;
            class_correct[actual_label]++;
        }
    }
    
    double final_accuracy = static_cast<double>(total_correct) / test_inputs.size();
    
    // --- 7. 输出测试结果 ---
    std::cout << "\n--- 测试集评估结果 ---" << std::endl;
    std::cout << " - 测试样本总数: " << test_inputs.size() << std::endl;
    std::cout << " - 正确预测数量: " << total_correct << std::endl;
    std::cout << " - 整体准确率: " << std::fixed << std::setprecision(2) << final_accuracy * 100 << "%" << std::endl;
    
    // 输出每个类别的准确率
    std::cout << "\n各数字类别识别准确率:" << std::endl;
    for (int i = 0; i < 10; i++) {
        double class_acc = (class_total[i] > 0) ? 
                          static_cast<double>(class_correct[i]) / class_total[i] * 100 : 0.0;
        
        std::cout << " - 数字 " << i << ": " 
                  << class_correct[i] << "/" << class_total[i] << " = "
                  << std::fixed << std::setprecision(2) << class_acc << "%" << std::endl;
    }

    // --- 8. 展示部分预测结果 ---
    std::cout << "\n--- 部分预测示例 ---" << std::endl;
    const int num_examples = 10;
    for (int i = 0; i < num_examples && i < test_inputs.size(); ++i) {
        std::vector<double> prediction = network.predict(test_inputs[i]);
        int predicted_label = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        int actual_label = std::distance(test_targets[i].begin(), std::max_element(test_targets[i].begin(), test_targets[i].end()));
        
        std::cout << "测试样本 " << i << ": ";
        if (predicted_label == actual_label) {
            std::cout << "正确! ";
        } else {
            std::cout << "错误! ";
        }
        std::cout << "预测 = " << predicted_label << ", 实际 = " << actual_label;
        
        // 显示预测置信度
        double confidence = *std::max_element(prediction.begin(), prediction.end()) * 100;
        std::cout << " | 置信度: " << std::fixed << std::setprecision(1) << confidence << "%" << std::endl;
    }

    std::cout << "\n按回车键退出程序...";
    std::cin.get();
    return 0;
}


// /*---------------------------------------------------*/
// //异或实验
// #include <iostream>
// #include <vector>
// #include <iomanip>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <chrono>
// #include <limits>
// #include <Windows.h>
// #include <cmath>
// #include "bpNet.h"

// // 增强的加载函数 (保持不变)
// bool loadDataFromFile(const std::string& filename, int num_inputs, int num_outputs,
//                       std::vector<std::vector<double>>& inputs, 
//                       std::vector<std::vector<double>>& targets);


// int main() {
//     SetConsoleOutputCP(65001);

//     // --- 1. 定义网络结构和数据文件 ---
//     const int input_nodes = 2;
//     const int hidden_nodes = 4;
//     const int output_nodes = 1;
//     const double learning_rate = 0.4;
//     const std::string train_filename = "data.txt";
//     const std::string test_filename = "test.txt";

//     // --- 2. 加载数据集 ---
//     std::vector<std::vector<double>> train_inputs, train_targets;
//     std::vector<std::vector<double>> test_inputs, test_targets;
    
//     std::cout << "--- 数据加载阶段 ---" << std::endl;
    
//     std::cout << "正在加载训练数据 (" << train_filename << ")..." << std::endl;
//     if (!loadDataFromFile(train_filename, input_nodes, output_nodes, train_inputs, train_targets)) {
//         std::cerr << "训练数据加载失败，程序即将退出。" << std::endl;
//         std::cin.get();
//         return 1;
//     }
    
//     std::cout << "正在加载测试数据 (" << test_filename << ")..." << std::endl;
//     if (!loadDataFromFile(test_filename, input_nodes, output_nodes, test_inputs, test_targets)) {
//         std::cerr << "测试数据加载失败，程序即将退出。" << std::endl;
//         std::cin.get();
//         return 1;
//     }
    
//     std::cout << "\n数据集统计信息:" << std::endl;
//     std::cout << " - 训练样本数: " << train_inputs.size() << std::endl;
//     std::cout << " - 测试样本数: " << test_inputs.size() << std::endl;
    
//     // --- 3. 获取用户输入参数 ---
//     double target_error;
//     int max_epochs;
//     std::cout << "\n--- 训练参数设置 ---" << std::endl;
//     std::cout << "请输入您的目标平均损失 (例如 0.001): ";
//     while (!(std::cin >> target_error) || target_error <= 0) {
//         std::cout << "输入无效，请输入一个大于0的数字: ";
//         std::cin.clear();
//         std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//     }
//     std::cout << "请输入最大训练轮数 (例如 50000): ";
//     while (!(std::cin >> max_epochs) || max_epochs <= 0) {
//         std::cout << "输入无效，请输入一个正整数: ";
//         std::cin.clear();
//         std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//     }

//     // --- 4. 创建网络并开始训练 ---
//     bpNet network(input_nodes, hidden_nodes, output_nodes, learning_rate);
//     std::cout << "\n--- 网络训练阶段 ---" << std::endl;
//     std::cout << "参数已设定。按回车键开始训练..." << std::endl;
//     std::cin.get();

//     auto start_time = std::chrono::high_resolution_clock::now();
//     bool target_achieved = false;
//     int final_epoch = 0;
//     double current_loss = 1.0;

//     for (int epoch = 1; epoch <= max_epochs; ++epoch) {
//         double epoch_loss = 0;
//         for (size_t i = 0; i < train_inputs.size(); ++i) {
//             network.train(train_inputs[i], train_targets[i]);
//             epoch_loss += network.get_loss(train_targets[i]);
//         }
//         current_loss = epoch_loss / train_inputs.size();
//         final_epoch = epoch;

//         if (epoch % 10 == 0 || epoch == 1 || epoch == max_epochs) {
//             std::cout << "\r训练进度: 轮次 " << std::setw(6) << epoch << "/" << max_epochs
//                       << " | 平均损失: " << std::fixed << std::setprecision(8) << current_loss
//                       << std::flush;
//         }

//         if (current_loss <= target_error) {
//             target_achieved = true;
//             break;
//         }
//     }
    
//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> training_duration = end_time - start_time;
//     std::cout << std::endl;

//     // --- 5. 输出最终训练结果 ---
//     std::cout << "\n--- 训练结果总结 ---" << std::endl;
//     if (target_achieved) std::cout << "训练成功！已达到目标误差。" << std::endl;
//     else std::cout << "训练停止。已达到最大训练轮数但未达到目标误差。" << std::endl;
//     std::cout << " - 目标误差: " << target_error << std::endl;
//     std::cout << " - 最终误差: " << current_loss << std::endl;
//     std::cout << " - 所用轮次: " << final_epoch << " / " << max_epochs << std::endl;
//     std::cout << " - 训练耗时: " << std::fixed << std::setprecision(3) << training_duration.count() << " 秒" << std::endl;

//     // --- 6. 在测试集上进行评估 ---
//     std::cout << "\n正在测试模型性能..." << std::endl;
    
//     int total_correct = 0;
//     // **【新功能】** 累加测试集的总误差
//     double total_test_error = 0.0;

//     for (size_t i = 0; i < test_inputs.size(); ++i) {
//         std::vector<double> prediction = network.predict(test_inputs[i]);
//         double raw_output = prediction[0];
//         double actual_target = test_targets[i][0];

//         // 计算准确率
//         int predicted_label = round(raw_output);
//         int actual_label = static_cast<int>(actual_target);
//         if (predicted_label == actual_label) {
//             total_correct++;
//         }
        
//         // 计算并累加均方误差
//         double error = raw_output - actual_target;
//         total_test_error += error * error;
//     }
    
//     double final_accuracy = (test_inputs.empty()) ? 0.0 : static_cast<double>(total_correct) / test_inputs.size();
//     // 计算平均测试误差
//     double average_test_error = (test_inputs.empty()) ? 0.0 : total_test_error / test_inputs.size();
    
//     // --- 7. 输出测试结果 ---
//     std::cout << "\n--- 测试集评估结果 ---" << std::endl;
//     std::cout << " - 测试样本总数: " << test_inputs.size() << std::endl;
//     std::cout << " - 正确预测数量: " << total_correct << std::endl;
//     std::cout << " - 整体准确率: " << std::fixed << std::setprecision(2) << final_accuracy * 100 << "%" << std::endl;
    
//     std::cout << " - 平均测试误差(MSE): " << std::fixed << std::setprecision(6) << average_test_error << std::endl;
    
//     // --- 8. 展示部分预测结果 ---
//     std::cout << "\n--- 部分预测示例 ---" << std::endl;
//     const int num_examples = 10;
//     for (int i = 0; i < num_examples && i < test_inputs.size(); ++i) {
//         std::vector<double> prediction = network.predict(test_inputs[i]);
//         double raw_output = prediction[0];
//         int predicted_label = round(raw_output);
//         int actual_label = static_cast<int>(test_targets[i][0]);
        
//         std::cout << "测试样本 " << i << " (输入:" << test_inputs[i][0] << "," << test_inputs[i][1] << "): ";
//         if (predicted_label == actual_label) std::cout << "正确! ";
//         else std::cout << "错误! ";
//         std::cout << "预测 = " << predicted_label << ", 实际 = " << actual_label;
        
//         double confidence = (predicted_label == 1) ? raw_output : (1.0 - raw_output);
//         std::cout << " | 置信度: " << std::fixed << std::setprecision(1) << confidence * 100 << "%";
        
//         std::cout << " (原始输出: " << std::setprecision(4) << raw_output << ")" << std::endl;
//     }

//     std::cout << "\n按回车键退出程序...";
//     std::cin.get();
//     return 0;
// }


// // 增强的加载函数 (保持不变)
// bool loadDataFromFile(const std::string& filename, int num_inputs, int num_outputs,
//                       std::vector<std::vector<double>>& inputs, 
//                       std::vector<std::vector<double>>& targets) {
//     inputs.clear();
//     targets.clear();
//     std::ifstream file(filename);

//     if (!file.is_open()) {
//         std::cerr << "错误：无法打开数据文件 \"" << filename << "\"。请检查文件是否存在于正确的位置。" << std::endl;
//         return false;
//     }

//     std::string line;
//     while (std::getline(file, line)) {
//         if (line.empty() || line[0] == '#') continue;

//         std::stringstream ss(line);
//         std::vector<double> input_sample;
//         std::vector<double> target_sample;
//         double value;

//         for (int i = 0; i < num_inputs; ++i) {
//             if (!(ss >> value)) {
//                 input_sample.clear(); 
//                 break;
//             }
//             input_sample.push_back(value);
//         }
        
//         if(input_sample.size() != num_inputs) continue;

//         for (int i = 0; i < num_outputs; ++i) {
//             if (!(ss >> value)) {
//                 target_sample.clear();
//                 break;
//             }
//             target_sample.push_back(value);
//         }

//         if (input_sample.size() == num_inputs && target_sample.size() == num_outputs) {
//             inputs.push_back(input_sample);
//             targets.push_back(target_sample);
//         }
//     }
//     file.close();

//     if (inputs.empty()) {
//         std::cerr << "警告：成功打开文件 \"" << filename << "\"，但未加载任何有效数据。" << std::endl;
//         std::cerr << "请检查文件内容格式是否正确（例如，每行应有 " << num_inputs + num_outputs << " 个由空格分隔的数字）。" << std::endl;
//         return false;
//     }
    
//     return true;

// }
