"""Compare YOLO26s-seg/640, YOLO26s-seg/960 and YOLO26m-seg/960 on the same still images."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
import torch

from scenelens import config
from scenelens.models.detector import Detector


VARIANTS = [('yolo26s-seg-640', 'yolo26s-seg.pt', 640),
            ('yolo26s-seg-960', 'yolo26s-seg.pt', 960),
            ('yolo26m-seg-960', 'yolo26m-seg.pt', 960)]


def sync():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def render_detections(image, detections):
    output = image.copy()
    for detection in detections:
        if detection.mask is not None:
            output[detection.mask] = (output[detection.mask]*.7 + np.array([0,220,120])*.3).astype(np.uint8)
    for detection in detections:
        x1, y1, x2, y2 = map(round, detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 120), 2)
        label = f'{detection.label} {detection.confidence:.0%}'
        cv2.putText(output, label, (max(0,x1), max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(output, label, (max(0,x1), max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0,255,120), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', type=Path)
    parser.add_argument('--device', default=config.DEFAULT_DEVICE, help='Default: CUDA with CPU fallback.')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--confidence', type=float, default=config.DETECTOR_CONFIDENCE)
    parser.add_argument('--output', type=Path, default=config.PROJECT_ROOT/'data/outputs/detection-comparison')
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or not 0 <= args.confidence <= 1:
        parser.error('Repeats/threads must be positive and confidence between 0 and 1.')
    torch.set_num_threads(args.threads)
    paths = args.images or [config.IMAGE_PATH]
    paths = [p for p in paths if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}]
    if not paths:
        parser.error('No supported input images found.')
    images = []
    for index, path in enumerate(paths,1):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Cannot read image: {path}')
        original_size = [image.shape[1],image.shape[0]]
        if config.IMAGE_MAX_WIDTH and image.shape[1] > config.IMAGE_MAX_WIDTH:
            scale=config.IMAGE_MAX_WIDTH/image.shape[1]
            image=cv2.resize(image,(config.IMAGE_MAX_WIDTH,max(1,round(image.shape[0]*scale))),interpolation=cv2.INTER_AREA)
        images.append((f'{index:02d}-{path.stem}',path,image,original_size))
    args.output.mkdir(parents=True,exist_ok=True)
    report = dict(confidence=args.confidence, repeats=args.repeats, threads=args.threads,
                  torch=torch.__version__, image_max_width=config.IMAGE_MAX_WIDTH,
                  scope='Detection and segmentation; latency excludes model load, image decoding and rendering. One warmup per image.',
                  accuracy='Unlabeled visual comparison; detection counts are not precision/recall or mAP.', variants=[])
    previews = {key:[] for key,_,_,_ in images}
    for name,weights,size in VARIANTS:
        entry=dict(name=name,weights=str(config.PROJECT_ROOT/'weights'/weights),image_size=size,images=[])
        report['variants'].append(entry)
        detector=None
        try:
            detector=Detector(entry['weights'],args.confidence,size,args.device,require_masks=True)
            entry['device']=detector.device
            gpu=detector.device != 'cpu' and torch.cuda.is_available()
            for key,path,image,original_size in images:
                detector.detect(image)  # Warm up this input shape.
                sync()
                if gpu:
                    torch.cuda.reset_peak_memory_stats()
                durations=[]
                for _ in range(args.repeats):
                    sync()
                    start=perf_counter()
                    detections=detector.detect(image)
                    sync()
                    durations.append((perf_counter()-start)*1000)
                overlay=args.output/f'{key}-{name}.png'
                rendered=render_detections(image,detections)
                if not cv2.imwrite(str(overlay),rendered):
                    raise RuntimeError(f'Cannot write overlay: {overlay}')
                entry['images'].append(dict(source=str(path),original_size=original_size,
                    analyzed_size=[image.shape[1],image.shape[0]],median_ms=median(durations),
                    timings_ms=durations,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if gpu else None,
                    counts=dict(Counter(d.label for d in detections)),detections=[dict(bbox=list(d.bbox),class_id=d.class_id,label=d.label,confidence=d.confidence,
                                     mask_pixels=int(d.mask.sum()) if d.mask is not None else None) for d in detections],overlay=str(overlay)))
                h,w=rendered.shape[:2]
                thumbnail=cv2.resize(rendered,(max(1,round(w*480/h)),480))
                thumbnail=cv2.copyMakeBorder(thumbnail,40,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
                cv2.putText(thumbnail,f'{name} | {len(detections)} objects',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                previews[key].append(thumbnail)
            entry['status']='complete'
        except (RuntimeError,OSError,ValueError) as error:
            entry['status']='failed'
            entry['error']=str(error)
        finally:
            del detector
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        print(f'{name}: {entry["status"]}',flush=True)
    for key,panels in previews.items():
        if panels:
            cv2.imwrite(str(args.output/f'{key}-comparison.png'),np.concatenate(panels,axis=1))
    report['complete']=all(v['status']=='complete' for v in report['variants'])
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Detection comparison','',report['accuracy'],'',report['scope'],'',
           '| Configuration | Image | Detections | Median ms | GPU allocation MB |','| --- | --- | ---: | ---: | ---: |']
    for variant in report['variants']:
        if variant['status']=='failed':
            lines.append(f'| {variant["name"]} | Unavailable: {variant["error"]} | — | — | — |')
        for result in variant['images']:
            memory=result['peak_gpu_allocated_mb']
            memory_text = f'{memory:.1f}' if memory is not None else 'N/A (CPU)'
            lines.append(f'| {variant["name"]} | {Path(result["source"]).name} | '
                         f'{len(result["detections"])} | {result["median_ms"]:.1f} | {memory_text} |')
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Results: {args.output}/report.md')
    return 0 if report['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
