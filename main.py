#!/bin/env python3 
''' main.py add distortions to a directory of images
Probably should use a multi-process implementation....
'''

import argparse
import json
import cv2
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path

from distortions import degradation_simulated

def simulate_distortion(input_file_path, output_path, args, config):
    
    imgs_params = [ ] 
    for i in range(args.repeat): 
        ori_img = cv2.imread(str(input_file_path)).astype(np.float32) / 255.0

        output_file_path = output_path / f'{input_file_path.stem}-{i:02d}.jpg'
        img_params = degradation_simulated(
                ori_img, save_to=str(output_file_path), **config)
        img_params['ref'] = input_file_path.name
        img_params['saved_name'] = output_file_path.name
        imgs_params.append(img_params)
        print(output_file_path)
    return imgs_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate image distortion')
    parser.add_argument('-i', '--input', type=str, required=True, help='directory to input image')
    parser.add_argument('--input_filelist', type=str, default='', help='directory to input filelist')
    parser.add_argument('--img_path_rel', action='store_true', help='if set, img path will be considered relative to input dir')
    parser.add_argument('-o', '--output', type=str, required=True, help='directory to output distorted image')
    parser.add_argument('-r', '--repeat', type=int, default=3, help='repetition for each image')
    parser.add_argument('-c', '--config', type=str, default='', help='config file')
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
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

    imgs_params = Parallel(n_jobs=-1)(delayed(simulate_distortion)(
        input_file_path, output_path, args, config
        ) for input_file_path in input_filelist)

    with open(output_path / 'res.json', 'w') as f: 
        json.dump(imgs_params, f)


