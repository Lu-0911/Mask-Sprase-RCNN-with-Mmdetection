_base_ = [
    '../_base_/datasets/voc_coco.py',
    '../_base_/schedules/schedule_1x.py', # 导入训练计划
    '../_base_/default_runtime.py' # 导入运行时配置
]

# 数据集路径和类别
data_root = 'D:/py/mmdetection/data/VOC_coco/'

# 修改所有 stage 的类别数
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(num_classes=20),
            dict(num_classes=20),
            dict(num_classes=20),
            dict(num_classes=20),
            dict(num_classes=20),
            dict(num_classes=20)
        ]
    )
)

# 学习率调整
optim_wrapper = dict(optimizer=dict(lr=1e-5))

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')