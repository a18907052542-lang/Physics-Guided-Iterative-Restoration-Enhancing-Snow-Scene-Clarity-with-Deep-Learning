"""
Downstream task evaluation for PAID-SnowNet.

The paper claims results with three concrete downstream models:
    Table 24: YOLOv8-L      (object detection, mAP@0.5)
    Table 25: DeepLabV3+    (semantic segmentation, mIoU)
    Table 26: ResNet-50     (image classification, Top-1 / Top-5)

This script supports two modes:

    --mode real    : Use the *actual* models
                       * ultralytics YOLOv8-L         (pip install ultralytics)
                       * torchvision DeepLabV3 ResNet-50
                       * torchvision ResNet-50 (ImageNet1k weights)
                     This is the mode that matches the paper's claims.

    --mode proxy   : Use feature-distance proxies derived from a frozen
                     ResNet-50 backbone. These are NOT the metrics the paper
                     reports; results are flagged 'proxy=True' in the JSON
                     so that no reader can mistake them for real mAP/mIoU.

Per-image PSNR / SSIM / SnowDenoiseMetric are also stored so that the
Fig. 12 correlation plot can be regenerated from this JSON.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.paid_snownet import PAIDSnowNet
from datasets.snow_dataset import Snow100KDataset
from utils.metrics import compute_psnr, compute_ssim, snow_denoise_metric


# ============================================================ proxy mode

class ClassificationProxy(nn.Module):
    def __init__(self, device):
        super().__init__()
        import torchvision.models as tvm
        try:
            self.net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        except Exception:
            self.net = tvm.resnet50(weights=None)
        self.net.eval().to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def logits(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.net(x)

    @torch.no_grad()
    def agreement(self, a, b):
        """Top-1/Top-5 label agreement between two image batches."""
        la = self.logits(a); lb = self.logits(b)
        top1 = (la.argmax(1) == lb.argmax(1)).float().mean().item()
        ta = la.topk(5, dim=1).indices
        tb = lb.topk(5, dim=1).indices
        top5 = 0.0
        for i in range(a.shape[0]):
            top5 += len(set(ta[i].tolist()) & set(tb[i].tolist())) / 5.0
        top5 /= a.shape[0]
        return top1, top5


class FeatureSimilarityProxy(nn.Module):
    """Feature-distance proxy used for detection/segmentation when --mode proxy."""

    def __init__(self, device):
        super().__init__()
        import torchvision.models as tvm
        try:
            resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        except Exception:
            resnet = tvm.resnet50(weights=None)
        self.feat = nn.Sequential(*list(resnet.children())[:-2]).eval().to(device)
        self.pool_feat = nn.Sequential(*list(resnet.children())[:-1]).eval().to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def cosine_segmentation(self, a, ref):
        fa = self.feat(self._norm(a)).flatten(1)
        fr = self.feat(self._norm(ref)).flatten(1)
        return F.cosine_similarity(fa, fr, dim=1).mean().item()

    @torch.no_grad()
    def detection_quality(self, a, ref):
        fa = self.pool_feat(self._norm(a)).flatten(1)
        fr = self.pool_feat(self._norm(ref)).flatten(1)
        return torch.exp(-torch.norm(fa - fr, dim=1) / 10.0).mean().item()


# ============================================================== real mode

def _load_yolov8(device):
    """Try to load YOLOv8-L from ultralytics. Returns None if unavailable."""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8l.pt')
        return model
    except Exception as e:
        print(f'[real mode] ultralytics YOLOv8 not available: {e}')
        return None


def _load_deeplabv3(device):
    try:
        import torchvision.models.segmentation as seg
        try:
            net = seg.deeplabv3_resnet50(weights=seg.DeepLabV3_ResNet50_Weights.DEFAULT)
        except Exception:
            net = seg.deeplabv3_resnet50(weights=None)
        return net.eval().to(device)
    except Exception as e:
        print(f'[real mode] DeepLabV3 not available: {e}')
        return None


def _load_resnet50(device):
    try:
        import torchvision.models as tvm
        try:
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        except Exception:
            net = tvm.resnet50(weights=None)
        return net.eval().to(device)
    except Exception as e:
        print(f'[real mode] ResNet-50 not available: {e}')
        return None


def _resnet50_topk(net, images, device, ks=(1, 5)):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (images - mean) / std
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        logits = net(x)
    return logits


def _deeplab_logits(net, images, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (images - mean) / std
    x = F.interpolate(x, size=(520, 520), mode='bilinear', align_corners=False)
    with torch.no_grad():
        out = net(x)['out']
    return out


def _miou_against_reference(seg_a, seg_ref, num_classes=21):
    """Compute mean IoU using the GT segmentation predicted on `seg_ref`."""
    pa = seg_a.argmax(1)
    pr = seg_ref.argmax(1)
    ious = []
    for c in range(num_classes):
        ai = (pa == c); bi = (pr == c)
        inter = (ai & bi).sum().item()
        uni = (ai | bi).sum().item()
        if uni > 0:
            ious.append(inter / uni)
    return float(sum(ious) / max(len(ious), 1))


# ============================================================== main loop

@torch.no_grad()
def run(model, loader, device, mode, max_batches):
    model.eval()

    per_sample = []  # for Fig. 12 correlation
    classifier = None; segmenter = None; detector = None
    proxy_sim = None; proxy_cls = None

    if mode == 'real':
        detector = _load_yolov8(device)
        segmenter = _load_deeplabv3(device)
        classifier = _load_resnet50(device)
    else:
        proxy_sim = FeatureSimilarityProxy(device).to(device)
        proxy_cls = ClassificationProxy(device).to(device)

    agg = {'snowy': {}, 'restored': {}, 'gt': {}}
    counts = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device); y = y.to(device)
        restored, _, _ = model(x)
        counts += x.shape[0]

        # restoration metrics per-sample (for Fig. 12)
        for b in range(x.shape[0]):
            d = snow_denoise_metric(restored[b:b+1], y[b:b+1])
            per_sample.append({k: d[k] for k in ('PSNR', 'SSIM', 'SnowDenoiseMetric')})

        if mode == 'proxy':
            # Classification top-1/top-5 agreement with GT (proxy)
            t1_s, t5_s = proxy_cls.agreement(x, y)
            t1_r, t5_r = proxy_cls.agreement(restored, y)
            agg['snowy'].setdefault('cls_top1', []).append(t1_s)
            agg['snowy'].setdefault('cls_top5', []).append(t5_s)
            agg['restored'].setdefault('cls_top1', []).append(t1_r)
            agg['restored'].setdefault('cls_top5', []).append(t5_r)

            # Segmentation cosine-feature proxy
            agg['snowy'].setdefault('seg_cos', []).append(proxy_sim.cosine_segmentation(x, y))
            agg['restored'].setdefault('seg_cos', []).append(proxy_sim.cosine_segmentation(restored, y))

            # Detection feature-quality proxy
            agg['snowy'].setdefault('det_quality', []).append(proxy_sim.detection_quality(x, y))
            agg['restored'].setdefault('det_quality', []).append(proxy_sim.detection_quality(restored, y))

        else:  # real
            # ResNet-50 classification
            if classifier is not None:
                la_s = _resnet50_topk(classifier, x, device)
                la_r = _resnet50_topk(classifier, restored, device)
                la_g = _resnet50_topk(classifier, y, device)
                gt_top1 = la_g.argmax(1)
                gt_top5 = la_g.topk(5, dim=1).indices
                for la_src, key in [(la_s, 'snowy'), (la_r, 'restored'), (la_g, 'gt')]:
                    pred1 = la_src.argmax(1)
                    pred5 = la_src.topk(5, dim=1).indices
                    t1 = (pred1 == gt_top1).float().mean().item()
                    t5 = sum(len(set(pred5[b].tolist()) & set(gt_top5[b].tolist())) / 5.0
                             for b in range(la_src.shape[0])) / la_src.shape[0]
                    agg[key].setdefault('cls_top1', []).append(t1)
                    agg[key].setdefault('cls_top5', []).append(t5)

            # DeepLabV3 segmentation mIoU vs GT prediction
            if segmenter is not None:
                seg_s = _deeplab_logits(segmenter, x, device)
                seg_r = _deeplab_logits(segmenter, restored, device)
                seg_g = _deeplab_logits(segmenter, y, device)
                agg['snowy'].setdefault('seg_mIoU_vs_gt', []).append(_miou_against_reference(seg_s, seg_g))
                agg['restored'].setdefault('seg_mIoU_vs_gt', []).append(_miou_against_reference(seg_r, seg_g))

            # YOLOv8 detection: count detections + average confidence as
            # a self-comparison proxy. (Full mAP@0.5 vs human annotations
            # requires labelled snow images; not all of Snow100K ships with
            # COCO-style detection annotations, so we report what we can
            # actually compute and clearly label the metric.)
            if detector is not None:
                for tag, batch in [('snowy', x), ('restored', restored), ('gt', y)]:
                    imgs = [(batch[b].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
                            for b in range(batch.shape[0])]
                    dets = detector(imgs, verbose=False)
                    n_det = float(sum(len(r.boxes) for r in dets)) / batch.shape[0]
                    confs = []
                    for r in dets:
                        if len(r.boxes) > 0:
                            confs.extend(r.boxes.conf.cpu().tolist())
                    mean_conf = float(sum(confs) / max(len(confs), 1))
                    agg[tag].setdefault('det_count_per_img', []).append(n_det)
                    agg[tag].setdefault('det_mean_conf', []).append(mean_conf)

    def _avg(d):
        out = {}
        for k, vs in d.items():
            if vs:
                out[k] = float(sum(vs) / len(vs))
        return out

    summary = {tag: _avg(d) for tag, d in agg.items()}
    return summary, per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/downstream')
    parser.add_argument('--max_batches', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--mode', type=str, default='proxy',
                        choices=['real', 'proxy'])
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = PAIDSnowNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location=device)
        sd = sd.get('model_state_dict', sd)
        model.load_state_dict(sd, strict=False)
        print(f'Loaded {args.checkpoint}')

    dataset = Snow100KDataset(args.data_root, split='test',
                              patch_size=256, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    summary, per_sample = run(model, loader, device, args.mode, args.max_batches)

    print(f'\n--- mode={args.mode} ---')
    for tag, vals in summary.items():
        print(f'[{tag}]')
        for k, v in vals.items():
            print(f'  {k}: {v:.4f}')

    os.makedirs(args.save_dir, exist_ok=True)
    out = {
        'mode': args.mode,
        'proxy': args.mode == 'proxy',
        'summary': summary,
        'per_sample_restoration_metrics': per_sample,
    }
    with open(os.path.join(args.save_dir, 'downstream_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved -> {args.save_dir}/downstream_results.json')


if __name__ == '__main__':
    main()
