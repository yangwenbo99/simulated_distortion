#!/bin/env python3 
''' main.py add distortions from https://github.com/yohan-pg/robust-unsupervised 
to a directory of images for benchmarking degradation synthesis methods. 

For each image x_0, the script will randomly select another image x_1.  
Then, two global distortion will be selected from "N", "A", or "NA" (or the
lack of), and two inpainting distortions (or the lack of) will be determined. 
Each image will be combined with the 2*2=4 distortions.


To show the effectiveness of local degradation encoding and synthesis, we 
need inpainting combined with either or both of noise and JPEG artifacts.

Same for compression
'''


import argparse
import json
from pathlib import Path
from typing import List, Dict
import random


import cv2
import numpy as np
from joblib import Parallel, delayed
import torch
import torchvision
import benchmark
from benchmark.degradations import cycle_to_file

## _TASKS[name][level] is a task
#_TASKS: Dict[str,List[benchmark.Task]] = []
_TASKS: Dict[str, benchmark.Task] = {}

def init_tasks():
    global _TASKS
    '''
    for name in benchmark.task_names:
        tasks = {}
        for level in benchmark.task_levels:
            tasks[level] = benchmark.get_task(name, level)
        _TASKS[name] = tasks
        '''
    _TASKS['P'] = benchmark.get_task('inpainting', 2)
    _TASKS['N'] = benchmark.get_task('denoising', 2)
    _TASKS['A'] = benchmark.get_task('deartifacting', 2)
    _TASKS['ASF'] = benchmark.get_task('deartifacting', 'L')
    _TASKS['NA'] = benchmark.Task(
            'NA',
            "composed_tasks",
            2,
            benchmark.init_composed(2),
            ['denoising', 'deartifacting'],
            )

class IdentityDegradation:
    def degrade_ground_truth(self, ground_truth, save_path=None):
        return ground_truth


def simulate_distortion(
        index, 
        input_file_path_0, input_file_path_1, input_file_path_2, 
        output_path, args, config):
    def _load_img(input_file_path):
        img = torchvision.io.read_image(str(input_file_path.absolute()), torchvision.io.ImageReadMode.RGB)
        img = img.float() / 255
        img.unsqueeze_(0)
        img = img.cuda()
        return img


    imgs_params = [ ] 

    if 'g_tasks' and 'l_tasks' in config:
        g_tasks = config['g_tasks']
        l_tasks = config['l_tasks']
    else:
        g_tasks = []
        l_tasks = []
        while len(g_tasks) < 2 or len(l_tasks) < 2:
            # Keep rolling the dice until
            # at least one local distortion and at least one global distortion
            # is selected, and both images will receive at least one distortion
            for _ in range(2):
                g_tasks.append(np.random.choice(['N', 'A', 'NA', None]))
            for _ in range(2):
                l_tasks.append(np.random.choice(['P', 'P', 'P', None]))
            if all(x is None for x in g_tasks):
                g_tasks = []
                l_tasks = []
            if all(x is None for x in l_tasks):
                g_tasks = []
                l_tasks = []
            for g, l in zip(g_tasks, l_tasks):
                if g is None and l is None:
                    g_tasks = []
                    l_tasks = []

    x0 = _load_img(input_file_path_0)
    x1 = _load_img(input_file_path_1)
    x2 = _load_img(input_file_path_2)

    g_degs = [_TASKS[g_task_name].init_degradation() 
              if g_task_name is not None else IdentityDegradation()
              for g_task_name in g_tasks]
    l_degs = [_TASKS[l_task_name].init_degradation() \
            if l_task_name is not None else IdentityDegradation()
              for l_task_name in l_tasks]

    for gi, (g_task_name, g_deg) in enumerate(zip(g_tasks, g_degs)):
        for li, (l_task_name, l_deg) in enumerate(zip(l_tasks, l_degs)):
            saved_names = []
            for imgi, img in enumerate([x0, x1, x2]):
                output_file_path = output_path / f'{index:05d}-{input_file_path_0.stem}-{input_file_path_1.stem}-{input_file_path_2.stem}-{gi}-{li}-{imgi}.png'
                saved_names.append(output_file_path.name)
                target_g = g_deg.degrade_ground_truth(img, save_path=None)
                target_full = l_deg.degrade_ground_truth(target_g, save_path=None)
                cycle_to_file(target_full, output_file_path)
                assert target_full.shape == img.shape
            this_img_params = {
                    'x0': str(input_file_path_0.relative_to(input_path)),
                    'x1': str(input_file_path_1.relative_to(input_path)),
                    'x2': str(input_file_path_2.relative_to(input_path)),
                    'global_dist': g_task_name,
                    'local_dist': l_task_name,
                    'saved_names': saved_names,
                    }
            print('||'.join(saved_names), g_task_name, l_task_name)
            imgs_params.append(this_img_params)
    # In addition, we generate one with only global distortions
    saved_names = []
    for imgi, img in enumerate([x0, x1, x2]):
        output_file_path = output_path / f'{index:05d}-{input_file_path_0.stem}-{input_file_path_1.stem}-{input_file_path_2.stem}-global-{imgi}.png'
        saved_names.append(output_file_path.name)
        target_g = g_degs[0].degrade_ground_truth(img, save_path=None)
        cycle_to_file(target_g, output_file_path)
        assert target_full.shape == img.shape
    imgs_params.append({
            'x0': str(input_file_path_0.relative_to(input_path)),
            'x1': str(input_file_path_1.relative_to(input_path)),
            'x2': str(input_file_path_2.relative_to(input_path)),
            'global_dist': g_tasks[0],
            'local_dist': None,
            'saved_names': saved_names,
            })
    
    return imgs_params

def derange(l: list):
    '''In place derangement of a list
    '''
    for i in range(len(l) - 1, 0, -1):
        j = np.random.randint(0, i)
        l[i], l[j] = l[j], l[i]

def get_shuffled_lists(l: list, num_images: int = -1):
    '''Return a derangement of a list
    '''
    # Why do we need a fancy algorithm.... we can just do whatever it takes...
    if 0 < num_images < len(l):
        l = l.copy()
        random.shuffle(l)
        l3 = l[:num_images]
    elif num_images > len(l):
        raise ValueError('num_images must be less than or equal to the length of the list')
    else:
        # num_images == len(l) or not supplied
        l3 = l
    l1 = []
    l2 = []
    for i, x in enumerate(l3):
        while ((j := np.random.randint(0, len(l))) == i):
            pass
        l1.append(l[j])
        while ((k := np.random.randint(0, len(l))) == i or k == j):
            pass
        l2.append(l[k])
    return l1, l2, l3




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate image distortion')
    parser.add_argument('-i', '--input', type=str, required=True, help='directory to input image')
    parser.add_argument('--input_filelist', type=str, default='', help='directory to input filelist')
    parser.add_argument('--img_path_rel', action='store_true', help='if set, img path will be considered relative to input dir')
    parser.add_argument('-o', '--output', type=str, required=True, help='directory to output distorted image')
    parser.add_argument('-c', '--config', type=str, default='', help='config file')  # If it ain't broke, don't fix it
    parser.add_argument('-n', '--n_jobs', type=int, default=-1, help='number of jobs to run in parallel')
    parser.add_argument('-N', '--n_images', type=int, default=-1, help='number of images to generate')

    args = parser.parse_args()

    init_tasks()

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

    # Generate another list of images to distort
    # No element should remain in the same position (we need derangements)
    input_filelist_1 = input_filelist.copy()
    # OK, let's implement Sattolo's algorithm
    # derange(input_filelist_1)
    input_filelist_1, input_filelist_2, input_filelist_3 = get_shuffled_lists(input_filelist, args.n_images)
    assert len(input_filelist_1) == args.n_images


    imgs_params = Parallel(n_jobs=args.n_jobs)(delayed(simulate_distortion)(
        index, input_file_path, input_filepath_1, input_filepath_2,
        output_path, args, config
        ) for index, (input_file_path, input_filepath_1, input_filepath_2) in enumerate(zip(input_filelist_1, input_filelist_2, input_filelist_3))
        )
    '''
    imgs_params = [simulate_distortion(
        input_file_path, output_path, args, config
        ) for input_file_path in input_filelist]
    '''

    with open(output_path / 'res.json', 'w') as f: 
        json.dump(imgs_params, f)


