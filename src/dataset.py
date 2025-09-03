import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

# Dataset class to handle pre, post, and mask images
class DamageDataset(Dataset):
    def __init__(self, pre_dir, post_dir, mask_dir, patch_size=128, stride=64, mode='post', balance = False, oversample=2):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.balance = balance
        self.oversample = oversample

        # Standard transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Collect image samples
        # Collect image samples
        self.filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(f"_{mode}_disaster_target.png")])
        self.samples = []
        patches_featuring_class = {'class0': 0, 'class1': 0, 'class2': 0, 'class3': 0, 'class4': 0}

        # first, count the patches with each class
        for fname in self.filenames:
            basename = fname.replace(f"_{mode}_disaster_target.png", "")
            mask = np.array(Image.open(os.path.join(self.mask_dir, fname)).convert('L'))
            h, w = mask.shape
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = mask[y:y + patch_size, x:x + patch_size]
                    #include = (4 in patch or 3 in patch or 2 in patch or np.random.rand() < 0.1)
                    #if include:
                    #    is_priority = any(cls in patch for cls in [2, 3, 4])
                    #    self.samples.append((basename, x, y, is_priority))
                    if True:
                        ###############################################################################################
                        has_c0, has_c1, has_c2, has_c3, has_c4 = (c in patch for c in [0, 1, 2, 3, 4])  # boolean
                        if has_c0:
                            patches_featuring_class[f'class0'] += 1
                        if has_c1:
                            patches_featuring_class[f'class1'] += 1
                        if has_c2:
                            patches_featuring_class[f'class2'] += 1
                        if has_c3:
                            patches_featuring_class[f'class3'] += 1
                        if has_c4:
                            patches_featuring_class[f'class4'] += 1

        for key, value in patches_featuring_class.items():
            print(f'\t{key} : {value}')
        print("-------------- All Samples Loaded --------------\n"
              "With oversampling/undersampling take this percentage of each class:\n")
        percent2include = {'class0': (oversample * patches_featuring_class[f'class4'] ),
                           'class1': (oversample * patches_featuring_class[f'class4']),
                           'class2': (oversample * patches_featuring_class[f'class4']),
                           'class3': (oversample * patches_featuring_class[f'class4']),
                           'class4': (patches_featuring_class[f'class4'] / patches_featuring_class[f'class0'])
                           }
        print(percent2include)
        ###############################################################################################
        """
        We are removing class 0 by 90%
        We need to oversmaple minority
        - Count amount of patches
        - Oversample number of patches and undersample majority
            - to do this take a random patch and make sure the name doesnt conflict (change 19th character)
            / option 1: load all samples; then go through samples randomly until quota is filled
            / option 2: as samples are loading; duplicate the samples as you are going
        """

    def oversampling(self, class0=1, class1=1, class2=1, class3=1, class4=1):
        for sample in self.samples:
            if 4 in sample:
                return None

        self.samples.append()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basename, x, y, is_priority = self.samples[idx]
        pre_img = np.array(Image.open(os.path.join(self.pre_dir, f"{basename}_pre_disaster.png")).convert('RGB'))
        post_img = np.array(Image.open(os.path.join(self.post_dir, f"{basename}_post_disaster.png")).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.mask_dir, f"{basename}_{self.mode}_disaster_target.png")).convert('L'))

        # Crop to patch
        pre_patch = pre_img[y:y + self.patch_size, x:x + self.patch_size]
        post_patch = post_img[y:y + self.patch_size, x:x + self.patch_size]
        mask_patch = mask[y:y + self.patch_size, x:x + self.patch_size]

        # Apply transforms
        transform = self.aug_transform if is_priority else self.base_transform
        pre_patch = transform(Image.fromarray(pre_patch))
        post_patch = transform(Image.fromarray(post_patch))

        return pre_patch, post_patch, torch.from_numpy(mask_patch).long(), f"{basename}_x{x}_y{y}"
