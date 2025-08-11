#include "MNISTLoader.h"
#include <fstream>
#include <stdexcept>  //异常处理库
#include <cstdint>  //C++11  定义了一组固定宽度的整数
#include <iostream>

int MNISTLoader::reverseInt(int i) {
    unsigned char c1,c2,c3,c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool MNISTLoader::load_images(const std::string& filename, std::vector<std::vector<double>>& images, int max_images) {
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "无法打开图像文件："<< filename <<std::endl;
        return false;
    }

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char*)&num_images, sizeof(magic_number));
    num_images = reverseInt(num_images);

    file.read((char*)&rows, sizeof(rows));
    rows = reverseInt(rows);

    file.read((char*)&cols, sizeof(cols));
    cols = reverseInt(cols);

    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }

    images.resize(num_images, std::vector<double>(rows * cols));
    
    for(int i = 0; i< num_images; ++i) {
        for(int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = pixel / 255.0;
        }
    }
    return true;
}

bool MNISTLoader::load_labels(const std::string& filename, std::vector<std::vector<double>>& labels, int max_labels) {
    std::ifstream file(filename,std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "无法打开标签文件: " << filename << std::endl;
        return false;
    }

    int magic_number= 0, num_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    
    file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverseInt(num_labels);

    if (max_labels > 0 && max_labels < num_labels) {
        num_labels = max_labels;
    }

    labels.resize(num_labels, std::vector<double>(10, 0.0));
    for(int i = 0; i< num_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label,sizeof(label));
        labels[i][label] = 1.0;  //on-hot编码  将离散的分类标签转换为二进制向量
    }
    return true;
}