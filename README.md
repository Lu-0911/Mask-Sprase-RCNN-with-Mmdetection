# Mask-Sprase-RCNN-with-Mmdetection

## 1. 项目概述

本项目旨在利用开源目标检测工具箱 MMDetection，在 Pascal VOC 数据集上实现 Mask R-CNN 和 Sparse R-CNN 模型的训练、评估和可视化。通过对比这两种模型在同一数据集上的表现，探索它们在目标检测和实例分割任务上的异同。

## 2. 环境准备

本项目依赖于 MMDetection 框架。

1.  **克隆 MMDetection 仓库：**
    ```bash
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    ```

2.  **创建并激活虚拟环境：**
    使用 Conda 创建一个干净的虚拟环境。
    ```bash
    conda create -n mmdet python=3.8 -y
    conda activate mmdet
    ```

3.  **安装 PyTorch 和 TorchVision：**
    请根据设备的 CUDA 版本选择合适的安装命令。例如，对于 CUDA 11.3：
    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```

4.  **安装 MMCV：**
    ```bash
    pip install -U openmim
    mim install mmcv-full
    ```

5.  **安装 MMDetection：**
    ```bash
    pip install -v -e .
    ```

## 3. 数据集准备

本项目使用 Pascal VOC 2007 和 2012 数据集进行训练和测试。

1.  **下载 VOC 数据集：**
    从 [Pascal VOC 官网]("http://host.robots.ox.ac.uk/pascal/VOC/") 下载 VOC2007 train/val, VOC2007 test 和 VOC2012 train/val 数据集。

2.  **组织数据集目录：**
    将下载的数据集放置在 MMDetection 项目的 `data` 目录下，并按照 MMDetection 的要求组织目录结构。通常结构如下：
    ```
    mmdetection
    ├── data
    │   ├── VOCdevkit
    │   │   ├── VOC2007
    │   │   │   ├── Annotations
    │   │   │   ├── ImageSets
    │   │   │   ├── JPEGImages
    │   │   │   └── SegmentationObject
    │   │   └── VOC2012
    │   │       ├── Annotations
    │   │       ├── ImageSets
    │   │       ├── JPEGImages
    │   │       └── SegmentationObject
    │   └── voc
    ```

3.  **转换为 COCO 格式 (可选但推荐)：**
    MMDetection 对 COCO 格式支持更好。您可以使用 MMDetection 提供的工具将 VOC 数据集转换为 COCO 格式。具体方法请参考 MMDetection 官方文档中关于数据集准备的部分。
    ```bash
    python tools/dataset_converters/pascal_voc.py data\VOCdevkit --out-dir data\VOC_coco --out-format coco
    ```

## 4. 模型配置

本项目使用 Mask R-CNN 和 Sparse R-CNN 模型。MMDetection 提供了基于 COCO 数据集的默认配置文件，这里的`voc_coco.py` 提供了数据集的加载配置

### 4.1 配置文件准备

1.  **复制配置文件：**
    复制配置文件（`mask_rcnn_r50_fpn_1x_voc.py` 和 `sparse-rcnn_r50_fpn_1x_voc.py`）到自定义的目录（例如 `configs/voc/`），可以添加修改。
    这里的`voc_coco.py` 提供了数据集的加载配置。

    在配置文件中，修改以下关键项：
    *   `num_classes`: 将类别数量从 80 (COCO) 修改为 20 (VOC)。
    *   `data_root` 和 数据集相关的路径：指向您的 VOC 数据集路径。
    *   `dataset_type`: 如果转换为 COCO 格式，可能需要修改为 `CocoDataset` 并调整相应的管道。
    *   `load_from`: 指定预训练权重文件的路径 (例如 `checkpoints/mask_rcnn_r50_fpn_1x_coco.pth`)。
    *   `work_dir`: 指定训练日志和权重文件的保存目录。

    示例修改后的 Mask R-CNN VOC 配置文件路径：`work_dirs/mask_rcnn_voc/mask_rcnn_r50_fpn_1x_voc.py`。

## 5. 模型训练

使用 MMDetection 提供的 `tools/train.py` 脚本进行模型训练。

```bash
# 训练 Mask R-CNN 在 VOC 数据集上
python tools/train.py configs/voc/mask_rcnn_r50_fpn_1x_voc.py

# 训练 Sparse R-CNN 在 VOC 数据集上
python tools/train.py configs/voc/sparse-rcnn_r50_fpn_1x_voc.py
```

训练过程中的日志和模型权重将保存在配置文件中指定的 work_dir 目录下。

## 6. 模型测试与评估

使用 MMDetection 提供的 tools/test.py 脚本对训练好的模型进行测试和评估。

```bash
# 测试 Mask R-CNN 模型
python tools/test.py configs/voc/mask_rcnn_r50_fpn_1x_voc.py work_dirs/mask_rcnn_voc/latest.pth --eval bbox segm

# 测试 Sparse R-CNN 模型
python tools/test.py configs/voc/sparse-rcnn_r50_fpn_1x_voc.py work_dirs/sparse_rcnn_voc/latest.pth --eval bbox segm
```

--eval bbox segm 参数表示评估目标检测 (bbox) 和实例分割 (segm) 的性能。测试结果将输出到终端，也可以通过 --out 参数指定输出文件。

## 7.日志分析与可视化
MMDetection 支持多种可视化后端，包括 TensorBoard。您可以在配置文件中设置 vis_backends 来启用 TensorBoard。

训练日志文件 ( .log ) 包含了训练过程中的详细信息。可以使用 tools/analysis_tools/analyze_logs.py 脚本分析日志并绘制曲线图。

```bash
# 绘制训练损失曲线 (以 Mask R-CNN 为例)
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/mask_rcnn_voc/20250526_114301/20250526_114301.log --keys loss --out work_dirs/mask_rcnn_voc/loss_curve.png

# 绘制 mAP 曲线 (如果日志中包含评估结果)
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/mask_rcnn_voc/20250526_114301/20250526_114301.log --keys bbox_mAP segm_mAP --out work_dirs/mask_rcnn_voc/map_curve.png
```
可以使用 tools/visualize_results.py 脚本将模型的检测结果可视化到图像上。

```bash
# 可视化 Mask R-CNN 在测试集上的结果
python tools/visualize_results.py configs/voc/mask_rcnn_r50_fpn_1x_voc.py work_dirs/mask_rcnn_voc/latest.pth --show --out-dir work_dirs/mask_rcnn_voc/visualization
```

## 8. 实验结果
实验结果包括训练过程中的损失曲线、评估指标 (如 mAP)、各类别性能对比以及可视化结果图。

* 训练日志和权重： 保存在 work_dirs 目录下，例如 work_dirs/mask_rcnn_voc/ 。
* TensorBoard 数据： 保存在 work_dirs 目录下相应实验的 vis_data 子目录中，例如 work_dirs/mask_rcnn_voc/vis_data/ 。
* 测试结果： 输出到终端或指定的文件。
* 可视化结果图： 保存在指定的输出目录，例如 work_dirs/mask_rcnn_voc/visualization/ 。
