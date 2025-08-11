#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>

class MNISTLoader {
public:
    static bool load_images(const std::string& filename, std::vector<std::vector<double>>& images, int max_images = -1);
    static bool load_labels(const std::string& filename, std::vector<std::vector<double>>& labels, int max_labels = -1);
private:
    static int reverseInt(int i);
};

#endif