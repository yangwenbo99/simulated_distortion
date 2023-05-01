# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os

# Define the custom dataset class
class DistortedImageDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, reverse_distorted=False):
        """
        Args:
            json_file (string): Path to the json file with image information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file, 'r') as f:
            self.image_info = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.reverse_distorted = reverse_distorted

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the given index.

        Args:
            idx (int): Index of the item to be retrieved.

        Returns:
            sample (dict): A dictionary containing the following keys:
                - 'ref_image': A list of tensors, each of shape (C, H, W), representing the reference images.
                - 'distorted_images': A list of lists of tensors, each of shape (C, H, W), representing the distorted images.
                - 'ssim_values': A list of lists of SSIM values between the reference and distorted images.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref_images = [ ] 
        distorted_images = [ ]
        ssim_values = [ ]
        for gidx, img_saves in enumerate(self.image_info[idx]['img_saves']): 
            # Load reference image
            ref_img_path = os.path.join(self.root_dir, img_saves['ref_name'])
            ref_image_this = Image.open(ref_img_path).convert('RGB')

            # Load distorted images
            distorted_images_this = []
            ssim_values_this = []
            for distorted_name, ssim_value in zip(img_saves['distorted_names'], img_saves['ssims']):
                distorted_img_path = os.path.join(self.root_dir, distorted_name)
                distorted_image = Image.open(distorted_img_path).convert('RGB')
                distorted_images_this.append(distorted_image)
                ssim_values_this.append(ssim_value)

            # Reverse the order of distorted images if reverse_distorted is True
            if self.reverse_distorted:
                distorted_images_this = list(reversed(distorted_images_this))
                ssim_values_this = list(reversed(ssim_values_this))

            # Apply transformations if any
            if self.transform:
                ref_image_this = self.transform(ref_image_this)
                distorted_images_this = [self.transform(distorted_image) for distorted_image in distorted_images_this]
            ref_images.append(ref_image_this)
            distorted_images.append(distorted_images_this)
            ssim_values.append(ssim_values_this)

        sample = {'ref_image': ref_images, 'distorted_images': distorted_images, 'ssim_values': ssim_values}

        return sample

def collate_fn(batch):
    """
    Custom collate function to handle variable number of distorted images per reference image.

    Args:
        batch (list): List of samples from the dataset.  Each sample is a dict
        containing the followings: 
            - 'ref_image': A list of tensors (len=gn), each of shape (C, H, W),
              representing the reference images.
            - 'distorted_images': A list (len=gn) of lists of tensors
              (len=gsz), each of shape (C, H, W), representing the distorted
              images.
            - 'ssim_values': A list (len=gn) of lists (len=gsz) of SSIM values
              between the reference and distorted images.

    Each sample is assumed to have the same size (H, W), same number of 
    groups (i.e. the same number of references, gn) and the same number of
    distorted images (gsz) . 

    Returns:
        collated_batch (dict): A dictionary containing the following keys:
            - 'ref_image': A tensor shaped (B, gn, c, H, W)
            - 'distorted_images': A tensor shaped (B, gn, gsz, C, H, W)
            - 'ssim_values': A tensor shaped (B, gn, gsz)
    """
    # print(batch[0]['ref_image'])
    ref_images = torch.stack([torch.stack(item['ref_image'], dim=0) for item in batch], dim=0)
    distorted_images = torch.stack(
            [
                torch.stack([
                    torch.stack(distorted_group, dim=0) for distorted_group in item['distorted_images']
                ], dim=0)
                for item in batch
            ], 
            dim=0)
    ssim_values = torch.tensor([item['ssim_values'] for item in batch])

    collated_batch = {'ref_image': ref_images, 'distorted_images': distorted_images, 'ssim_values': ssim_values}

    return collated_batch

if __name__ == '__main__': 
    # Define the transformations
    data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Create the dataset and dataloader
    distorted_dataset = DistortedImageDataset(
            json_file='/mnt/data/datasets/wm_simulated/distance_learning/g1g2_test/res.json', root_dir='/mnt/data/datasets/wm_simulated/distance_learning/g1g2_test', transform=data_transform)
    dataloader = DataLoader(distorted_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for xx in dataloader: 
        print(xx['ref_image'].shape)
        print(xx['distorted_images'].shape)
        print(xx['ssim_values'].shape)
        break
