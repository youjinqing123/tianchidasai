import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/retinanet_r50_fpn_1x.py')
cfg.model.pretrained = None

# 构建网络，载入模型
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)

#_ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
# 如果通过网盘下载，取消下一行代码的注释，并且注释掉上一行
_ = load_checkpoint(model, 'pths/epoch_20_SGD_1e-3_1235.pth')

# 测试一张图片
img = mmcv.imread('250.jpg')
result = inference_detector(model, img, cfg)
print(result)

#show_result(img, result)


# 测试多张图片
'''
imgs = ['0101.jpg', '0102.jpg', 'n1.jpg','n2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result)
'''
