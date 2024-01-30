#!/bin/python3 
''' main.py add distortions to a directory of images
Probably should use a multi-process implementation....
'''
import argparse
import json
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate image distortion')
    parser.add_argument('-i', '--input', type=str, required=True, help='directory to the json file')
    parser.add_argument('-o', '--output', type=str, required=True, help='directory to the output image list file')
    parser.add_argument(
            '--output-train',  dest='output_train', type=str, 
            default='', help='directory to the output train image list file')
    parser.add_argument(
            '--output-test',  dest='output_test', type=str, 
            default='', help='directory to the output test image list file')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    with open(args.output, 'w', newline='') as f:
        '''Each row has the follwoing contents
        Some of them can be None. 

            parameters['focus'] = {
                    'method': 'gaussian', 
                    'kernel_size': int(kernel_size), 
                    'radius': float(radius)
                    }
            parameters['motion'] = {
                    'method': 'simple', 
                    'angle': int(angle), 
                    'kernel_size': int(kernel_size)
                    }
            parameters['exposure'] = {
                    'alpha': float(alpha), 
                    'beta': float(beta)
                    }
            parameters['noise'] = {
                    'flicker_strength': flicker_strength, 
                    'gaussian_strength': gauss_strength, 
                    }
                    
            parameters['jpeg'] = {
                    'quality': int(quality)
                    }
            parameters['ref'] = ref_name
            parameters['saved_name'] = name
        '''
        fieldnames = [
                'name', 'ref_name', 
                'focus_method', 'focus_kernel_size', 'focus_radius', 
                'motion_method', 'motion_angle', 'motion_kernel_size', 
                'exposure_alpha', 'exposure_beta',
                'flicker_strength', 'gauss_strength', 'jpeg_quality'
                ] 
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for parametergs in data: 
            for parameters in parametergs: 
                writer.writerow({
                    'name': parameters['saved_name'],
                    'ref_name': parameters['ref'],
                    'focus_method': parameters['focus']['method'] if parameters['focus'] is not None else None,
                    'focus_kernel_size': parameters['focus']['kernel_size'] if parameters['focus'] is not None else -1,
                    'focus_radius': parameters['focus']['radius'] if parameters['focus'] is not None else -1,
                    'motion_method': parameters['motion']['method'] if parameters['motion'] is not None else None,
                    'motion_angle': parameters['motion']['angle'] if parameters['motion'] is not None else -1,
                    'motion_kernel_size': parameters['motion']['kernel_size'] if parameters['motion'] is not None else -1,
                    'exposure_alpha': parameters['exposure']['alpha'] if parameters['exposure'] is not None else -1,
                    'exposure_beta': parameters['exposure']['beta'] if parameters['exposure'] is not None else -1,
                    'flicker_strength': parameters['noise']['flicker_strength'],
                    'gauss_strength': parameters['noise']['gaussian_strength'], 
                    'jpeg_quality': parameters['jpeg']['quality'] if parameters['jpeg'] is not None else -1,
                })

    # Now, do a 80/20 split of the generated csv for trainig and testing
    # if the two arguments are supplied
    total_len = len(data) * len(parametergs)
    if args.output_train and args.output_test:
        with open(args.output, 'r') as f:
            reader = csv.DictReader(f)
            train_writer = csv.DictWriter(open(args.output_train, 'w'), fieldnames=fieldnames)
            test_writer = csv.DictWriter(open(args.output_test, 'w'), fieldnames=fieldnames)
            train_writer.writeheader()
            test_writer.writeheader()
            for i, row in enumerate(reader):
                if i >= total_len * 0.8:
                    test_writer.writerow(row)
                else:
                    train_writer.writerow(row)
