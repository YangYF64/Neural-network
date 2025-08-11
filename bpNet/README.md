# 基于C++的BP神经网络 🧠

## ⚠️ 
本项目**不包含 MNIST 数据集文件**，您必须按照以下步骤自行下载和配置。

---

### 1. 访问官网
前往 MNIST 官方网站：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### 2. 下载文件
请下载页面中的 **全部四个** 数据文件：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

### 3. 解压文件
使用您的解压工具（如 **7-Zip**, **WinRAR** 等）解压下载的 `.gz` 文件，您将得到四个原始的 `.idx` 文件。

### 4. 放置文件
将这四个解压后的文件，直接放入项目根目录下的 `data` 文件夹中即可。

---

✅ 完成后，您的 `data` 文件夹结构应如下所示：

```text
└── data/
    ├── t10k-images.idx3-ubyte
    ├── t10k-labels.idx1-ubyte
    ├── train-images.idx3-ubyte
    └── train-labels.idx1-ubyte
```

