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
    parser.add_argument('-r', '--root_name', type=str, default='images1024x1024_dist', help='root directory\'s name')
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
        ''' parameters['ref'] = ref_name
            parameters['saved_name'] = name
        '''
        fieldnames = [
                'name', 'ref_name', 
                ] 
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for parametergs in data: 
            for parameters in parametergs: 
                if 'ref-' not in parameters['ref']:
                    ref_name = 'images1024x1024/' + parameters['ref']
                else:
                    ref_name = args.root_name + '/' + parameters['ref']
                writer.writerow({
                    'name': args.root_name + '/' + parameters['saved_name'],
                    'ref_name': ref_name, # 'images1024x1024/' + parameters['ref'],
                })

    # Now, do a 80/20 split of the generated csv for trainig and testing
    # if the two arguments are supplied
    total_len = len(data) 
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
