#!/bin/env python3 
''' main_wm_benchmark.py Generate an extra test set from the Wikimedia dataset. 

The training and test set consist of 50k images.  We can start from the 
(50k+1)st image. 

One degradation will be selected for each pair of images. 
'''

import argparse
import json
import cv2
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
import random

from distortions import degradation_simulated, degradation_simulated_deterministic

def simulate_distortion(
        serial: int,
        input_file_1_path: Path,
        input_file_2_path: Path,
        output_path, args, config):
    
    imgs_params = [ ] 
    for i in range(args.repeat): 
        ori_img_1 = cv2.imread(str(input_file_1_path)).astype(np.float32) / 255.0
        ori_img_2 = cv2.imread(str(input_file_2_path)).astype(np.float32) / 255.0

        output_file_1_path = output_path / f'{serial:05d}-{input_file_1_path.stem}-{input_file_2_path.stem}-a-{i:02d}.jpg'
        output_file_2_path = output_path / f'{serial:05d}-{input_file_1_path.stem}-{input_file_2_path.stem}-b-{i:02d}.jpg'
        img_params = degradation_simulated(
                ori_img_1, save_to=str(output_file_1_path), **config)
        degradation_simulated_deterministic(
                ori_img_2, parameters=img_params, 
                save_to=str(output_file_2_path))
        img_params['ref_1'] = input_file_1_path.name
        img_params['saved_name_1'] = output_file_1_path.name
        img_params['ref_2'] = input_file_2_path.name
        img_params['saved_name_2'] = output_file_2_path.name
        imgs_params.append(img_params)
        print(output_file_1_path, '|', output_file_2_path)
    return imgs_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate image distortion')
    parser.add_argument('-i', '--input', type=str, required=True, help='directory to input image')
    parser.add_argument('--input_filelist', type=str, default='', help='directory to input filelist')
    parser.add_argument('--img_path_rel', action='store_true', help='if set, img path will be considered relative to input dir')
    parser.add_argument('-o', '--output', type=str, required=True, help='directory to output distorted image')
    parser.add_argument('-r', '--repeat', type=int, default=3, help='repetition for each image')
    parser.add_argument('-c', '--config', type=str, default='', help='config file')
    parser.add_argument('-j', '--n_jobs', type=int, default=-1, help='number of jobs to run in parallel')
    parser.add_argument('-n', '--n_pairs', type=int, default=-1, help='number of image pairs, cannot be greater than the number of images in the input directory / 2')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
            config = {}
    if args.input_filelist:
        with open(args.input_filelist, 'r') as f:
            input_filelist = f.read().splitlines()
        if args.img_path_rel: 
            input_filelist = [input_path / file for file in input_filelist]
        else: 
            input_filelist = [Path(file) for file in input_filelist]
    else:
        input_filelist = list(sorted(input_path.glob('*')))

    print(f'Got {len(input_filelist)} Images')


    if args.n_pairs < 0:
        args.n_pairs = len(input_filelist) // 2
    random.shuffle(input_filelist)
    first_half = input_filelist[:args.n_pairs]
    second_half = input_filelist[args.n_pairs:2*args.n_pairs]


    imgs_params = Parallel(n_jobs=args.n_jobs)(delayed(simulate_distortion)(
        serial, input_file_1_path, input_file_2_path,
        output_path, args, config
        ) for serial, (input_file_1_path, input_file_2_path) in enumerate(zip(first_half, second_half))
        )

    with open(output_path / 'res.json', 'w') as f: 
        json.dump(imgs_params, f)
