#!/bin/python3 
''' main_groupped.py add distortions to a directory of images

Example of execution:
    python main_grouped.py -i input_images/ -o output_images/ --config config.json --distortions focus,motion,noise -n 8 -N 8
'''

import argparse
import shutil
import json
import cv2
import numpy as np
from typing import List
from joblib import Parallel, delayed
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

import distortions
from distortions import degradation_simulated

def simulate_distortion(
        input_file_paths: List[Path], 
        output_path: Path, 
        args: argparse.Namespace, 
        config: dict):
    '''    
    Simulates distortion on a list of input images and saves the distorted images to the output path.

    Args:
        input_file_paths (List[Path]): List of input image file paths.
        output_path (Path): Path to save the distorted images.
        args (argparse.Namespace): Parsed command line arguments.
        config (dict): Configuration dictionary.

    Returns:
        res (dict): A dictionary containing the following keys:
            - 'dist_params': List of distortion parameters used for each distorted image.
            - 'img_saves': List of dictionaries, each containing:
                - 'ref_name': Name of the reference image.
                - 'distorted_names': List of names of the distorted images.
                - 'ssims': List of SSIM values between the reference and distorted images.
    '''

    dist_params_normalized = distortions.random_parameter_group(
            distortions=args.distortions, 
            n=args.num_parameter_group, ordered=args.ordered_distortion
            )
    dist_params = [distortions.denormalize_parameters(x) for x in dist_params_normalized]
    
    img_saves = [ ]
    for input_idx, input_file_path in enumerate(input_file_paths): 
        input_duplic_file_path = output_path / input_file_path.name
        shutil.copyfile(input_file_path, input_duplic_file_path)

        ori_img = cv2.imread(str(input_file_path)).astype(np.float32) / 255.0

        img_saves_this_ref = {
                'ref_name': str(input_duplic_file_path.name),
                'distorted_names': [ ],  
                'ssims': [ ] 
                }

        for dist_idx, dist_param in enumerate(dist_params): 
            output_file_path = output_path / f'{input_file_path.stem}-{dist_idx:02d}.jpg'
            distortions.degradation_simulated_deterministic(
                    ori_img, parameters=dist_param, save_to=str(output_file_path), **config)
            img_saves_this_ref['distorted_names'].append(output_file_path.name)

            # Calculate SSIM for the distorted image
            distorted_img = cv2.imread(str(output_file_path)).astype(np.float32) / 255.0
            #print(ori_img.shape, distorted_img.shape)
            ssim_value = ssim(ori_img, distorted_img, data_range=1, channel_axis=2)
            img_saves_this_ref['ssims'].append(float(ssim_value))

            print(output_file_path)

        img_saves.append(img_saves_this_ref)

    if args.ordered_ssim: 
        assert len(input_file_paths) == 1, 'When ordered_ssim is set, group_size must be 1'
        # Sort distorted_names and dist_params by ssim
        sorted_indices = np.argsort(img_saves_this_ref['ssims'])
        img_saves_this_ref['distorted_names'] = [img_saves_this_ref['distorted_names'][i] for i in sorted_indices]
        img_saves_this_ref['ssims'] = [img_saves_this_ref['ssims'][i] for i in sorted_indices]
        dist_params = [dist_params[i] for i in sorted_indices]

    res = {
            'dist_params_normalized': dist_params_normalized,
            'dist_params': dist_params,
            'img_saves': img_saves
            }
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate image distortion')
    parser.add_argument('-i', '--input', type=str, required=True, help='directory to input image')
    parser.add_argument('--input_filelist', type=str, default='', help='directory to input filelist')
    parser.add_argument('--img_path_rel', action='store_true', help='if set, img path will be considered relative to input dir')
    parser.add_argument('-o', '--output', type=str, required=True, help='directory to output distorted image')
    parser.add_argument('-c', '--config', type=str, default='', help='config file')

    parser.add_argument('--distortions', type=str, default='focus,motion,noise', 
                        help='types of distortions, can be focus, motion, exposure, noise and jpeg')
    parser.add_argument(
            '-n', '--num_parameter_group', type=int, default=8, 
            help='number of distorted images for one reference image')
    parser.add_argument(
            '-N', '--group_size', type=int, default=8, 
            help='number reference in one group')
    parser.add_argument('--same_ref', action='store_true', 
                        help='if set, the same distortion group will have the same refererence image')
    parser.add_argument('--ordered_distortion', action='store_true', 
                        help='if set, distortions will ordered')
    parser.add_argument('--ordered_ssim', action='store_true', 
                        help='if set, distorted images will be ordered by their SSIM')
    parser.add_argument('--indent', type=int, default=0, 
                        help='if set, the JSON will be formatted')
    args = parser.parse_args()

    args.distortions = args.distortions.split(',')

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
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
    
    if not args.same_ref: 
        # Group input_filelist into groups sized args.group_size
        input_filelist_groups = [input_filelist[i:i + args.group_size] for i in range(0, len(input_filelist), args.group_size)]
    else: 
        input_filelist_groups = [[input_file] * args.group_size for input_file in input_filelist]


    # Update the loop to iterate over input_filelist_groups
    imgs_params = Parallel(n_jobs=-1)(delayed(simulate_distortion)(input_file_group, output_path, args, config) for input_file_group in input_filelist_groups)
    # imgs_params = [simulate_distortion(input_file_group, output_path, args, config) for input_file_group in input_filelist_groups]

    with open(output_path / 'res.json', 'w') as f: 
        if args.indent: 
            json.dump(imgs_params, f, indent=args.indent)
        else: 
            json.dump(imgs_params, f)
