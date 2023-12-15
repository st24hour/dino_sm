from torch.utils.data import Dataset
import os
import glob
import imageio
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

import openslide
from torchvision import transforms
import random
from PIL import Image, ImageFilter, ImageOps
import h5py

class Patches_Dataset(Dataset):
    def __init__(self, file_paths, wsis, custom_transforms, pretrained, target_patch_size=-1, split='train'):
        """
        Args:
        file_path (string): List of paths to the .h5 file containing patched data.
        wsis: List of wsi file paths
        pretrained (bool): Use ImageNet transforms
        custom_transforms (callable, optional): Optional transform to be applied on a sample
        target_patch_size (int): Custom defined image size before embedding
        """
        self.split = split
        self.file_pathes = file_paths
        self.wsis = wsis
        if not custom_transforms:
            self.roi_transforms = ValTransform()
        else:
            self.roi_transforms = custom_transforms

        self.dset = np.array([], dtype=np.int64).reshape(0,2) #empty array
        self.coord_num_list = [] #accumulated number of batches
        for idx, h5_file_path in enumerate(self.file_pathes):
            with h5py.File(h5_file_path, "r") as f:
                self.dset = np.vstack([self.dset, np.array(f['coords'])])
                self.coord_num_list.append(len(self.dset))
                self.patch_level = f['coords'].attrs['patch_level']
                self.patch_size = f['coords'].attrs['patch_size']
                #print("id:", idx, "patch_size:", self.patch_size)
                if target_patch_size > 0:
                    self.target_patch_size = (target_patch_size,) * 2
                else:
                    self.target_patch_size = None
            #if idx==1:
            #    break
        self.length = len(self.dset)

    def __len__(self):
        if self.split == 'train':
            return self.length
        return 1

    def __getitem__(self, idx):
        item = self.get_slide_id(idx=idx)
        wsi = openslide.open_slide(self.wsis[item])
        img = wsi.read_region(location=self.dset[idx], level=self.patch_level, size=(self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img)#.unsqueeze(0)
        return img

    def get_slide_id(self, idx):
        for i in range(0, len(self.coord_num_list)):
            if self.coord_num_list[i]>idx:
                item = i
                break
        return item
    
class ImageDataset(Dataset):
    def __init__(self, root, split='train'):
        self.split = split
        self.image_paths = []
        # TODO: save image paths to file to avoid reading overhead
        stamp_idxs = sorted(os.listdir(root))
        for stamp_idx in stamp_idxs:
            image_paths = glob.glob(os.path.join(root, f'{stamp_idx}/[0-9]*.png'))
            image_paths = sorted(filter(lambda x: not 'key' in x, image_paths))
            self.image_paths += image_paths

    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths)
        return 1

    def __getitem__(self, idx):
        if self.split != 'train': # randomly choose an image for validation
            idx = np.random.choice(len(self.image_paths), 1)[0]
        image = imageio.imread(self.image_paths[idx])
        if image.shape[-1] == 4: # if there is alpha channel
            image[image[..., -1]==0, :3] = 255 # a=0 to white
            image = image[..., :3]
        return self.transform(image)

# class ValTransform:
#     def __init__(self):
#         self.norm = A.Compose([
#             A.Resize(224, 224, interpolation=cv2.INTER_AREA),
#             A.Normalize(),
#             ToTensorV2()
#         ])

#         self.orig = A.Compose([
#             A.Resize(224, 224, interpolation=cv2.INTER_AREA),
#             ToTensorV2()
#         ])

#     def __call__(self, image):
#         return [self.orig(image=image)['image'], self.norm(image=image)['image']]
class ValTransform:
    def __init__(self):
        self.norm = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.orig = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __call__(self, image):
        return [self.orig(image), self.norm(image)]

# class TrainTransform:
#     def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
#         flip_and_color_jitter = A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
#             A.ToGray(p=0.2)
#         ])
#         normalize = A.Compose([
#             A.Normalize(),
#             ToTensorV2()
#         ])

#         # area interpolation should be better for small image
#         # first global crop
#         self.global_crop1 = A.Compose([
#             A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_AREA),
#             flip_and_color_jitter,
#             A.GaussianBlur(p=1.0),
#             normalize,
#         ])
#         # second global crop
#         self.global_crop2 = A.Compose([
#             A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_AREA),
#             flip_and_color_jitter,
#             A.GaussianBlur(p=0.1),
#             A.Solarize(p=0.2),
#             normalize,
#         ])
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_crop = A.Compose([
#             A.RandomResizedCrop(96, 96, scale=local_crops_scale, interpolation=cv2.INTER_AREA),
#             flip_and_color_jitter,
#             A.GaussianBlur(p=0.5),
#             normalize,
#         ])

#     def __call__(self, image):
#         crops = [self.global_crop1(image=image)['image'], self.global_crop2(image=image)['image']]
#         for _ in range(self.local_crops_number):
#             crops += [self.local_crop(image=image)['image']]
#         return crops
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #original dino vit
            #transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #for vit-l/14 from openclip laion2b
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img