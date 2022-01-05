from argparse import ArgumentParser

import torch
from mmcv.parallel import collate, scatter
from mmseg.apis import show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmseg.core.evaluation import get_palette
from mmseg.datasets.pipelines import Compose, MultiScaleFlipAug
from mmseg.models import BaseSegmentor
from mmseg.ops import resize

from segformer.model import segformer_b2_city


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    model = segformer_b2_city(pretrained=True, progress=True)
    model.eval()
    model.to(device)

    data_pipeline = Compose([
        LoadImage(),
        MultiScaleFlipAug(
            img_scale=(2048, 512),
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ],
        ),
    ])

    data = data_pipeline({'img': args.img})
    data = collate([data], samples_per_gpu=1)

    if device.type == 'cuda':
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    seg_logit = model(data['img'][0])
    size = data['img_metas'][0][0]['ori_shape'][:2]
    seg_logit = resize(
        seg_logit,
        size=size,
        mode='bilinear',
        align_corners=model.align_corners,
        warning=False)
    seg_pred = seg_logit.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()
    result = list(seg_pred)

    model.CLASSES = ['placeholder'] * model.decode_head.num_classes
    model.show_result = BaseSegmentor.show_result.__get__(model, None)
    show_result_pyplot(model, args.img, result, get_palette('cityscapes'))


if __name__ == '__main__':
    main()
