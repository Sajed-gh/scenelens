"""Evaluate metric depth on timestamp-associated TUM RGB-D pairs (no scale fitting)."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from scenelens import config
from scenelens.models.depth import DepthEstimator


def depth_metrics(prediction, truth, max_depth):
    if prediction.shape != truth.shape or truth.ndim != 2:
        raise ValueError('Prediction and sensor depth must have matching 2D shapes.')
    sensor_valid = np.isfinite(truth) & (truth > 0) & (truth <= max_depth)
    valid = sensor_valid & np.isfinite(prediction) & (prediction > 0)
    p, g = prediction[valid].astype(np.float64), truth[valid].astype(np.float64)
    error = np.abs(p - g)
    return dict(pixels=int(valid.sum()), sensor_pixels=int(sensor_valid.sum()),
                abs_sum=float(error.sum()), squared_sum=float(np.square(error).sum()),
                relative_sum=float((error / g).sum()), delta1_count=int((np.maximum(p / g, g / p) < 1.25).sum()))


def summarize(totals):
    count = totals['pixels']
    if not count:
        raise ValueError('No valid sensor/prediction pairs to evaluate.')
    return dict(pixels=count, sensor_pixels=totals['sensor_pixels'],
                prediction_coverage=count / totals['sensor_pixels'],
                mae_m=totals['abs_sum'] / count, rmse_m=(totals['squared_sum'] / count)**.5,
                abs_rel=totals['relative_sum'] / count, delta1=totals['delta1_count'] / count)


def associated_pairs(path, max_difference):
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f'Expected RGB timestamp/path and depth timestamp/path at line {line_number}.')
        rgb_time, rgb_path, depth_time, depth_path = fields
        if abs(float(rgb_time) - float(depth_time)) > max_difference:
            continue
        yield rgb_time, rgb_path, depth_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tum-root', required=True, type=Path)
    parser.add_argument('--associations', type=Path, help='Defaults to TUM root/associations.txt')
    parser.add_argument('--max-pairs', type=int, default=100)
    parser.add_argument('--max-difference', type=float, default=.02)
    parser.add_argument('--max-depth', type=float, default=20.)
    parser.add_argument('--depth-scale', type=float, default=5000.)
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--output', type=Path, default=Path('data/outputs/depth_evaluation.json'))
    args = parser.parse_args()
    if args.max_pairs < 1 or args.max_depth <= 0 or args.depth_scale <= 0 or args.max_difference < 0:
        parser.error('Evaluation counts/scales must be positive and timestamp tolerance nonnegative.')
    if args.threads:
        torch.set_num_threads(args.threads)
    pairs = list(associated_pairs(args.associations or args.tum_root / 'associations.txt', args.max_difference))[:args.max_pairs]
    if not pairs:
        raise ValueError('No timestamp-associated RGB/depth pairs found.')
    estimator = DepthEstimator(config.DEPTH_MODEL, config.DEPTH_CACHE_DIR, args.device, config.DEPTH_INPUT_SIZE)
    rows = []
    totals = dict(pixels=0, sensor_pixels=0, abs_sum=0., squared_sum=0., relative_sum=0., delta1_count=0)
    for index, (timestamp, rgb_path, depth_path) in enumerate(pairs):
        rgb = cv2.imread(str(args.tum_root / rgb_path))
        sensor = cv2.imread(str(args.tum_root / depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or sensor is None:
            raise ValueError(f'Cannot read evaluation pair: {rgb_path}, {depth_path}')
        if sensor.dtype != np.uint16 or sensor.ndim != 2 or sensor.shape != rgb.shape[:2]:
            raise ValueError('Expected registered 16-bit TUM depth matching RGB dimensions.')
        prediction = estimator.estimate(rgb)
        values = depth_metrics(prediction, sensor.astype(np.float32) / args.depth_scale, args.max_depth)
        for key in totals:
            totals[key] += values[key]
        rows.append(dict(timestamp=timestamp, rgb=rgb_path, depth=depth_path,
                         **(summarize(values) if values['pixels'] else dict(pixels=0))))
    report = dict(dataset=str(args.tum_root), model=config.DEPTH_MODEL, device=estimator.device,
                  evaluated_pairs=len(rows), max_depth_m=args.max_depth, depth_scale=args.depth_scale,
                  max_timestamp_difference_s=args.max_difference, scale_alignment=False,
                  metrics=summarize(totals))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as output:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None
