''' lmdb_pack.py pack images into an lmdb database. 

By default, only the image name (without parents) is used as the key. 
'''
import os
import time
import argparse
import lmdb
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def main(args: argparse.Namespace):

    df = pd.read_csv(args.file_list)
    img_list = [Path(os.path.join(args.reference_dir, row['ref_name'])) for _, row in df.iterrows()]
    img_list += [Path(os.path.join(args.distorted_dir, row['name'])) for _, row in df.iterrows()]
    img_list = list(sorted(list(set(img_list))))
    
    total_len = len(img_list)
    print(f"Found {total_len} items")

    start = time.time()
    print("Time elapsed: %.2fs / %d samples" % ((time.time() - start), total_len))

    lmdb_path = args.output_path
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    lmdb_envd = lmdb.open(os.path.join(lmdb_path, 'data'), map_size=2**37)
    with lmdb_envd.begin(write=True) as lmdb_txn:
        for i in tqdm(range(total_len)):
            img_path = img_list[i]
            with open(img_path, 'rb') as f:
                binary_data = f.read()
            lmdb_txn.put(img_path.name.encode(), binary_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack data into lmdb')
    # parser.add_argument('--csv_path', type=str, help='path to csv file')
    parser.add_argument('-o', '--output_path', type=str, help='path to output lmdb file')
    parser.add_argument('-r', '--reference_dir', type=str, help='directory containing reference images')
    parser.add_argument('-d', '--distorted_dir', type=str, help='directory containing distorted images')
    parser.add_argument('-f', '--file_list', type=str, help='list of input image files (CSV)')


    args = parser.parse_args()
    main(args)

