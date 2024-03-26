#!/bin/env python3 
''' main.py add distortions from https://github.com/yohan-pg/robust-unsupervised 
to a directory of images
Probably should use a multi-process implementation....
'''

import argparse
import json
import cv2
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
import torch
import torchvision
import benchmark
from benchmark.degradations import cycle_to_file

def simulate_distortion(input_file_path, output_path, args, config):
    
    imgs_params = [ ] 
    for i in range(args.repeat): 
        ori_img = torchvision.io.read_image(str(input_file_path.absolute()), torchvision.io.ImageReadMode.RGB)
        ori_img = ori_img.float() / 255
        ori_img.unsqueeze_(0)
        ori_img = ori_img.cuda()
        output_file_path = output_path / f'{input_file_path.stem}-{i:02d}.jpg'

        task_id = np.random.randint(len(tasks))
        task = tasks[task_id]
        degradation = task.init_degradation()
        target = degradation.degrade_ground_truth(ori_img, save_path=output_file_path)
        if target.shape != ori_img.shape:
            ref_img = torch.nn.functional.interpolate(ori_img, size=target.shape[-2:], mode='area')
            output_ref_file_path = output_path / f'ref-{input_file_path.stem}-{i:02d}.png'
            cycle_to_file(ref_img, output_ref_file_path)
        else:
            output_ref_file_path = input_file_path

        img_params = { 'distortion_id': int(task_id) }
        img_params['ori'] = input_file_path.name
        img_params['ref'] = output_ref_file_path.name
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
    parser.add_argument('--resolution', type=int, default=1024, help='The resolution of generated image') # Actually not used
    parser.add_argument('-c', '--config', type=str, default='', help='config file')
    
    args = parser.parse_args()

    tasks = benchmark.all_tasks

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

    print(f'Got {len(input_filelist)} Images')

    imgs_params = Parallel(n_jobs=-1)(delayed(simulate_distortion)(
        input_file_path, output_path, args, config
        ) for input_file_path in input_filelist)

    with open(output_path / 'res.json', 'w') as f: 
        json.dump(imgs_params, f)
