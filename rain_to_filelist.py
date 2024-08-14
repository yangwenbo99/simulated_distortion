#!/usr/bin/env python

'''rain_to_filelist.py: This script is used to for processing the list of file
in dataset given by the following paper to a consist format of my projects:

    T. Wang, X. Xin, K. Xu, S. Chen, Q Zhang and R. W. H. Lau, "Spatial 
    Attentive Single-Image Deraining with a High Quality Real Rain Dataset",
    in CVPR, 2019.

This is not a standard part of this distortion simulation project, but I also
make use of the dataset.

Basic information about the dataset's structure:

- distorted patch: `./real_world/000/000-f/000-f_x_y.png`
- Corresponding gt: `./real_world_gt/000/000_x_y.png`

where
- f: Frame number
- x,y: Position in frame
- 000: video index
'''

import argparse
import csv
from  pathlib import Path
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a randomized list of image patches from a video dataset')
    parser.add_argument('-n', '--num-frames', type=int, required=True, help='number of frames to select per video index and position')
    parser.add_argument('-o', '--output', type=str, required=True, help='output CSV file')
    parser.add_argument('-i', '--input', type=str, required=True, help='input root path for the dataset')
    parser.add_argument('-c', '--check', action='store_true', help='check if the is readable')
    args = parser.parse_args()

    if args.check:
        from PIL import Image

    gt_dir = Path(args.input) / 'real_world_gt'
    distorted_dir = Path(args.input) / 'real_world'

    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['name', 'ref_name', 'video_index', 'frame', 'x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for gt_video_dir_path in sorted(gt_dir.iterdir()):
            # `real_world_gt/000/`
            video_index = gt_video_dir_path.name
            distorted_video_dir_path = distorted_dir / video_index

            dist_video_dir_path = distorted_dir / video_index
            dist_frame_dir_paths = sorted(list(dist_video_dir_path.iterdir()))
            num_frames = len(dist_frame_dir_paths)

            # Due to the phisical procedure of generating the dataset 
            # (see their paper for more details), the number of distorted
            # frames variies from patch to patch.
            available_frames_for_patches = { }
            for gt_patch_path in sorted(gt_video_dir_path.iterdir()):
                _, x, y = gt_patch_path.stem.split('_')
                x, y = int(x), int(y)
                available_frames_for_patches[(x, y)] = []
            for dist_frame_dir_path in dist_frame_dir_paths:
                # `real_world/000/000-f`
                f = dist_frame_dir_path.name.split('-')[1]
                for dist_patch_path in sorted(dist_frame_dir_path.iterdir()):
                    _, x, y = dist_patch_path.stem.split('_')
                    x, y = int(x), int(y)
                    available_frames_for_patches[(x, y)].append(f)
            # print(f'{len(available_frames_for_patches)} patches in {video_index}')

            for gt_patch_path in sorted(gt_video_dir_path.iterdir()):
                # `real_world_gt/000/000_x_y.png`

                if args.check:
                    # Check if the file is readable as an image by PIL
                    try:
                        Image.open(gt_patch_path)
                    except Exception as e:
                        print(f'Error reading {gt_patch_path}: {e}')
                        continue

                _, x, y = gt_patch_path.stem.split('_')
                x, y = int(x), int(y)
                available_frames = available_frames_for_patches[(x, y)]
                num_sampled = min(args.num_frames, len(available_frames))
                selected_frames = random.sample(range(len(available_frames)), num_sampled)
                for frame_idx_a in selected_frames:
                    # The index in available_frames

                    # Coreesponding to 000
                    f = available_frames[frame_idx_a] 
                    if f == '14' and video_index == '274':
                        continue # This frame's dir structure is wrong

                    frame_dir_path = dist_video_dir_path / f'{video_index}-{f}'
                    # `real_world/000/000_-f`
                    dist_patch_path = frame_dir_path / f'{video_index}-{f}_{x}_{y}.png'
                    assert frame_dir_path.is_dir(), f'{frame_dir_path} does not exist'
                    assert dist_patch_path.is_file(), f'{dist_patch_path} does not exist'


                    if args.check:
                        # Check if the file is readable as an image by PIL
                        try:
                            Image.open(dist_patch_path)
                        except Exception as e:
                            print(f'Error reading {dist_patch_path}: {e}')
                            continue

                    writer.writerow({
                        'name': str(dist_patch_path),
                        'ref_name': str(gt_patch_path),
                        'video_index': video_index,
                        'frame': f,
                        'x': x,
                        'y': y
                    })



