import cv2
from pathlib import Path
from mmdet.apis import init_detector, inference_detector

def visualize(model, image_path, output_path):
    result = inference_detector(model, image_path)
    img = model.show_result(
        image_path,
        result,
        score_thr=0.5,
        show=False,
        bbox_color=(255, 0, 0),
        text_color=(200, 200, 200),
        mask_color=(0, 255, 0)
    )
    cv2.imwrite(output_path, img)

# 加载模型
mask_rcnn = init_detector(
    'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc.py',
    'work_dirs/mask_rcnn_voc/best_bbox_mAP_epoch_12.pth',
    device='cuda:0'
)
sparse_rcnn = init_detector(
    'configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_voc.py',
    'work_dirs/sparse_rcnn_voc/best_bbox_mAP_epoch_12.pth',
    device='cuda:0'
)

# 可视化测试集图像（4张）
test_images = [
    'data/VOC_coco/images/000001.jpg',
    'data/VOC_coco/images/000002.jpg',
    'data/VOC_coco/images/000003.jpg',
    'data/VOC_coco/images/000004.jpg'
]
for img_path in test_images:
    base_name = Path(img_path).name
    visualize(mask_rcnn, img_path, f'output/mask_rcnn_{base_name}')
    visualize(sparse_rcnn, img_path, f'output/sparse_rcnn_{base_name}')

# 可视化外部图像（3张）
external_images = ['external1.jpg', 'external2.jpg', 'external3.jpg']
for img_path in external_images:
    base_name = Path(img_path).name
    visualize(mask_rcnn, img_path, f'output/mask_rcnn_external_{base_name}')
    visualize(sparse_rcnn, img_path, f'output/sparse_rcnn_external_{base_name}')