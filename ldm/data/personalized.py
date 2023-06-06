import json

from typing import OrderedDict
from os.path import join, splitext
from random import Random, SystemRandom

import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import functional as TVF
from captionizer import caption_from_path, generic_captions_from_path, find_images

VALID_CROP_TYPES = [
    'center', 'random'
]

class CenterCropFraction(torch.nn.Module):
    def __init__(self, frac):
        super().__init__()
        assert frac < 1.0
        self.frac = frac

    def forward(self, img, min_size_pixels=None):
        min_size_pixels = int(min_size_pixels) if min_size_pixels is not None else None

        if isinstance(img, torch.Tensor):
            width, height = img.shape[-1], img.shape[-2]
        elif isinstance(img, Image.Image):
            width, height = img.size
        else:
            raise Exception(f'Unsupported image type {type(img).__name__}')

        if width < height:
            # Width shortest
            new_width, new_height = self._get_new_dims(width, height, min_size_pixels=min_size_pixels)
        else:
            # Height shortest
            new_height, new_width = self._get_new_dims(height, width, min_size_pixels=min_size_pixels)

        return TVF.center_crop(img, (new_height, new_width))

    def _get_new_dims(self, short_dim, long_dim, min_size_pixels=None):
        new_short_dim = round(short_dim * self.frac)
        new_short_dim = (max(min_size_pixels, new_short_dim)
                        if min_size_pixels is not None else new_short_dim)

        updated_frac = new_short_dim / short_dim
        new_long_dim = round(long_dim * updated_frac)
        # The following should not be needed, but including it in case of any numerical shenanigans.
        new_long_dim = (max(min_size_pixels, new_long_dim)
                        if min_size_pixels is not None else new_long_dim)
        return new_short_dim, new_long_dim

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frac={self.frac})"

class UpscaleBase(Dataset):
    def __init__(self, *args) -> None:
        pass

class ConcatPersonalizedBase(Dataset):
    def __init__(self, *args,
                 add_flipped_frac=0.0,
                 add_sharpened_frac=0.0,
                 add_autocontrast_frac=0.0,
                 add_center_crop_frac=0.0,
                 add_rand_rotation_frac=0.0,
                 provide_negatives=False,
                 **kwargs):
        size = kwargs.get('size')
        center_crop_frac = 0.8
        center_crop_min_size = round(size / center_crop_frac) + 5

        self.provide_negatives = provide_negatives

        self.datasets = [
            PersonalizedBase(
                *args, sample_frac=1.0, image_transform=None, provide_negatives=provide_negatives, **kwargs),
            PersonalizedBase(
                *args, sample_frac=add_flipped_frac, provide_negatives=provide_negatives,
                image_transform=self._get_all_trf(transforms.RandomHorizontalFlip(p=1.0)), **kwargs),
            PersonalizedBase(
                *args, sample_frac=add_sharpened_frac, provide_negatives=provide_negatives,
                image_transform=self._get_all_trf(transforms.RandomAdjustSharpness(2, p=1.0)), **kwargs),
            PersonalizedBase(
                *args, sample_frac=add_autocontrast_frac, provide_negatives=provide_negatives,
                image_transform=self._get_all_trf(transforms.RandomAutocontrast(p=1.0)), **kwargs),
            PersonalizedBase(
                *args, sample_frac=add_center_crop_frac, provide_negatives=provide_negatives,
                image_transform=self._get_all_trf(CenterCropFraction(center_crop_frac)), apply_transform_last=False,
                min_size=center_crop_min_size, **kwargs),
            # RandomRotation doesn't work well – doesn't expand like expected
            PersonalizedBase(
                *args, sample_frac=add_rand_rotation_frac, provide_negatives=provide_negatives,
                image_transform=self._get_all_trf(transforms.RandomRotation(30, expand=True, interpolation=transforms.InterpolationMode.BILINEAR)),
                apply_transform_last=False, **kwargs
            )
        ]

        self._ds_idx_pairs = []
        for ds_idx in range(len(self.datasets)):
            for idx in range(len(self.datasets[ds_idx])):
                self._ds_idx_pairs.append((ds_idx, idx))

        shuffle_gen = Random(789)
        self._ds_idx_pairs = shuffle_gen.sample(
            self._ds_idx_pairs, k=len(self._ds_idx_pairs))

        self._length = len(self._ds_idx_pairs)

    def _get_all_trf(self, trf):
        if self.provide_negatives:
            trf = [trf] if trf is not None else []
            return transforms.Compose([
                *trf,
                transforms.RandomChoice([
                    #transforms.RandomPosterize(p=1.0, bits=2),
                    #transforms.RandomSolarize(p=1.0, threshold=160),
                    #transforms.ColorJitter(brightness=(1.50, 2.0)),
                    #transforms.ColorJitter(brightness=(0.05, 0.50)),
                    transforms.GaussianBlur(kernel_size=(15, 15), sigma=(2.5, 10.0)),
                    transforms.RandomAdjustSharpness(sharpness_factor=8, p=1.0)
                ])
            ])
        else:
            return trf

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        ds_idx, idx = self._ds_idx_pairs[idx]
        return self.datasets[ds_idx][idx]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 sample_frac=1.0,
                 image_transform=None,
                 apply_transform_last=True,
                 size=None,
                 min_size=None,
                 repeats=1,
                 interpolation='bicubic',
                 set='train',  # train or val (reg set is "train")
                 placeholder_token=None,
                 crop_type=None,
                 coarse_class_text=None,
                 token_only=False,
                 reg=False,
                 shuffle=True,
                 random_selection=False,
                 use_caption_file=False,
                 log_dir=None,
                 seed=None,
                 provide_negatives=False,
                 random_resize_to_min=None
                 ):

        self.set = set
        self.data_root = data_root

        image_paths = []
        for pth in list(find_images(self.data_root)):
            valid_suffixes = ['.png', '.jpg', '.jpeg']

            _, ext = splitext(pth)
            if ext.lower() not in valid_suffixes:
                if ext.lower() != '.json':
                    print(f'WARNING: Found non-image file "{pth}" in data root')
                continue

            if min_size is not None:
                # Skip too small images
                opened_img = Image.open(pth)
                h, w = opened_img.size
                if h < min_size or w < min_size:
                    continue

            image_paths.append(pth)

        if set == 'train' and seed is None:
            print('Note: seed not set in train set, generating random seed')

        self.seed = seed if seed is not None else SystemRandom().randint(1e6, 1e8)

        image_paths = Random(self.seed).sample(image_paths, k=int(sample_frac * len(image_paths)))
        if shuffle:
            image_paths = Random(self.seed + 1).sample(image_paths, k=len(image_paths))
        self.image_paths = image_paths

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.log_dir = log_dir

        self.use_caption_file = use_caption_file
        if self.use_caption_file:
            self.full_captions = {}
            captions_filename = join(self.data_root, 'captions.json')

            with open(captions_filename, 'r') as f:
                caption_dict = json.load(f)

            for image_filename in self.image_paths:
                if image_filename not in caption_dict:
                    print(f'WARNING: Caption for "{image_filename}" not found in captions.json')
                else:
                    self.full_captions[image_filename] = caption_dict[image_filename]
        else:
            self.full_captions = None

        self.crop_type = crop_type
        if self.crop_type is not None and self.crop_type not in VALID_CROP_TYPES:
            raise Exception(f'crop_type must be None or one of {", ".join(VALID_CROP_TYPES)}')

        self.placeholder_token = placeholder_token
        self.token_only = token_only
        self.random_selection = random_selection

        self.coarse_class_text = coarse_class_text

        if self.use_caption_file and self.coarse_class_text is not None:
            print('WARNING: Class word will not be used as use_caption_file is enabled')

        if not self.placeholder_token and not self.use_caption_file:
            raise Exception('Must specify token or set use_caption_file to True')

        self._length = self.num_images * repeats

        self.size = size
        self.random_resize_to_min = random_resize_to_min

        self.randrange_gen = Random(self.seed + 2)
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]

        self.transform = image_transform
        self.apply_transform_last = apply_transform_last

        self.reg = reg
        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

        self.provide_negatives = provide_negatives

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        i = self.randrange_gen.randrange(0, self.num_images) if self.random_selection else i
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.transform is not None and not self.apply_transform_last:
            image = self.transform(image)

        example['caption'] = ''
        if self.use_caption_file:
            example['caption'] = self.full_captions.get(image_path, '')
        elif self.reg and self.coarse_class_text:
            example["caption"] = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            example["caption"] = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)

        if self.provide_negatives:
            example['caption'] = f'Bad quality, poor quality, blurry, jpeg artifacts. {example["caption"]}'

        if len(example['caption']) == 0:
            print(f'WARNING: Empty caption for image "{image_path}"')
            if self.log_dir is not None:
                with open(join(self.log_dir, f'{self.set}_missing_caption.txt'), 'a+') as f:
                    f.write(f'{image_path}\n')

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.random_resize_to_min:
            rand_size = round(self.randrange_gen.randrange(
                self.random_resize_to_min, self.size + 1) / 64) * 64
            target_size = (rand_size, rand_size)
        else:
            target_size = (self.size, self.size)

        if img.shape != target_size:
            if self.crop_type == 'center':
                crop = min(img.shape[0], img.shape[1])
                h, w = img.shape[0], img.shape[1]
                img = img[(h - crop) // 2:(h + crop) // 2,
                        (w - crop) // 2:(w + crop) // 2]
            elif self.crop_type == 'random':
                h, w = img.shape[0], img.shape[1]
                min_dim = min(h, w)
                new_h = np.random.randint(0, h - min_dim + 1)
                new_w = np.random.randint(0, w - min_dim + 1)
                img = img[new_h:new_h + min_dim, new_w:new_w + min_dim]

        image = Image.fromarray(img)
        if self.size is not None and image.size != target_size:
            image = image.resize(target_size,
                                 resample=self.interpolation)

        if self.transform is not None and self.apply_transform_last:
            image = self.transform(image)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
