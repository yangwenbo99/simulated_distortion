import argparse
import io
import time

import lmdb
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange


class ImageDatasetLMDB():

    def __init__(self, csv_path, db_dir):
        self.im_list = pd.read_csv(csv_path, header=0)

        db = lmdb.open(db_dir, readonly=True)
        self.txn = db.begin(write=False)

        self.transform = transforms.Compose([
            transforms.CenterCrop(50), 
            transforms.ToTensor()])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        imgbuf = self.txn.get(str(self.im_list['name'][idx]).encode())
        img = Image.open(io.BytesIO(imgbuf))
        sample = {'img': self.transform(img)}
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--db_dir', type=str, help='path to the lmdb database')
    parser.add_argument('--csv_path', type=str, help='path to the csv file')
    args = parser.parse_args()

    db_dir = args.db_dir
    csv_path = args.csv_path

    dataset = ImageDatasetLMDB(csv_path, db_dir=db_dir)
    total_len = len(dataset)

    train_loader = DataLoader(dataset, batch_size=400, shuffle=False, pin_memory=True, num_workers=8)

    start = time.time()
    with trange(len(train_loader)) as pbar:
        for step, sample_batched in enumerate(train_loader):
            im_batch = sample_batched['img']
            pbar.update()
    print("Time elapsed: %.2fs / %d samples" % ((time.time() - start), total_len))
