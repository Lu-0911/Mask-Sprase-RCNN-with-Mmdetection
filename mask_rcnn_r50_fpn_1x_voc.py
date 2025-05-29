_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py', # Mask R-CNN 模型配置
    '../_base_/datasets/voc_coco.py',  # COCO 格式配置
    '../_base_/schedules/schedule_1x.py', # 导入训练计划
    '../_base_/default_runtime.py' # 导入运行时配置
]

# 数据集路径和类别
data_root = '替换为实际目录/data/VOC_coco/'


# 修改模型输出类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

# 学习率调整
optim_wrapper = dict(optimizer=dict(lr=0.0025))

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')
