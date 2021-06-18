#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import config as cfg

from datasets.preprocess import coco_extract

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=True, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = cfg.OPENPOSE_PATH

    if args.train_files:
        # COCO dataset prepreocessing
        coco_extract(cfg.COCO_ROOT, openpose_path, out_path)

    if args.eval_files:
        print(f'not implemented yet!')
        # COCO dataset prepreocessing
        coco_extract(cfg.COCO_ROOT, openpose_path, out_path)
