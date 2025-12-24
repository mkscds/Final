import os
import random
import numpy as np
import pandas as pd
import gdown, zipfile
from PIL import Image
from pathlib import Path


import sklearn
import albumentations
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def download_isic_dataset(download_dir):

    os.makedirs(download_dir, exist_ok=True)
    extract_to = os.path.join(download_dir, "ISIC_2019")

    if os.path.exists(extract_to):
        print('Data already downloaded')
        return extract_to
    
    print('Downloading data')

    file_id = '15uNMnR3IoZaQYPoIAHJJeEfgACM5RBH3'
    filename = "ISIC_2019.zip"
    output = os.path.join(download_dir, filename)
    url = f'https://drive.google.com/uc?id={file_id}'

    print(f'Downloading file from {url} to {output}')
    gdown.download(url, output, quiet=False)

    if not os.path.exists(output):
        raise FileNotFoundError(f'Failed to download the file. Path "{output}" does not exist.')
    
    os.makedirs(extract_to, exist_ok=True)

    try:
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f'File extracted to {extract_to}')

    except:
        raise zipfile.BadZipFile('The download file is not a zip file or it is corrupted')
    
    return extract_to

class Isic2019raw(torch.utils.data.Dataset):

    dataset_downloaded = False

    def __init__(self, 
                 X_dtype=torch.float32, 
                 y_dtype=torch.int64, 
                 augmentations=None, 
                 data_path=None, 
                 vlm_prompt=True,
                 clip_tokenizer=None):

        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        if not Isic2019raw.dataset_downloaded:
            self.input_path = download_isic_dataset(data_path)  # your function
            Isic2019raw.dataset_downloaded = True
        else:
            self.input_path = os.path.join(data_path, "ISIC_2019")

        if not os.path.exists(self.input_path):
            raise ValueError(f"Dataset directory does not exist: {self.input_path}")

        self.idx2label = {
            0: "melanoma",
            1: "melanocytic nevus",
            2: "basal cell carcinoma",
            3: "actinic keratosis",
            4: "benign keratosis-like lesion",
            5: "dermatofibroma",
            6: "vascular lesion",
            7: "squamous cell carcinoma",
        }

        self.dic = {
            "input_preprocessed": os.path.join(self.input_path, "ISIC_2019_Training_Input_preprocessed"),
            "train_test_split": os.path.join(self.input_path, "train_test_split")
        }

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.vlm_prompt = vlm_prompt
        self.clip_tokenizer = clip_tokenizer

        df = pd.read_csv(self.dic["train_test_split"])
        self.df = df.copy()

        # Build text prompts
        prompt_template = "A clinical dermoscopy photograph of {}."
        prompt_list = [prompt_template.format(self.idx2label[i]) for i in range(len(self.idx2label))]

        self.prompts = prompt_list

        if self.clip_tokenizer is not None:
            # token_ids = self.clip_tokenizer(prompt_list, truncate=True)
            token_ids = self.clip_tokenizer(prompt_list)

            # handle MobileCLIP tensor case safely
            if isinstance(token_ids, torch.Tensor):
                self.text_tokens = token_ids.detach().clone()
            else:
                self.text_tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            self.text_tokens = None

        self.image_paths = [
            os.path.join(self.dic["input_preprocessed"], img + ".jpg")
            for img in self.df.image.tolist()
        ]

        self.targets = self.df.target.values
        self.centers = self.df.center.values
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = np.array(Image.open(self.image_paths[idx]))
        target = self.targets[idx]

        if self.augmentations is not None:
            img = self.augmentations(image=img)["image"]

        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.tensor(img, dtype=self.X_dtype)

        if self.vlm_prompt and self.text_tokens is not None:
            text_enc = self.text_tokens[target]  # 1D tensor of token ids
            return img, torch.tensor(target, dtype=self.y_dtype), text_enc

        return img, torch.tensor(target, dtype=self.y_dtype)
    

class FedIsic2019(Isic2019raw):
    def __init__(self,
                 train=True,
                 X_dtype=torch.float32,
                 y_dtype=torch.int64,
                 data_path=None,
                 cid=0,
                 pooled=False,
                 sz=224,
                 clip_tokenizer=None):

        if train:
            augmentations = albumentations.Compose([
                albumentations.RandomScale(0.07),
                albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1),
                albumentations.RandomCrop(sz, sz),
                albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                albumentations.Normalize(always_apply=True)
            ])
        else:
            augmentations = albumentations.Compose([
                albumentations.CenterCrop(sz, sz),
                albumentations.Normalize(always_apply=True)
            ])

        super().__init__(X_dtype=X_dtype, y_dtype=y_dtype, augmentations=augmentations,
                         data_path=data_path, clip_tokenizer=clip_tokenizer)

        self.train_test = "train" if train else "test"
        self.cid = cid
        self.pooled = pooled

        df = pd.read_csv(self.dic["train_test_split"])

        if pooled:
            df2 = df[df["fold"] == self.train_test].reset_index(drop=True)
        else:
            df2 = df[(df["fold"] == self.train_test) & (df["center"] == self.cid)].reset_index(drop=True)

        self.image_paths = [
            os.path.join(self.dic["input_preprocessed"], img + ".jpg")
            for img in df2.image.tolist()
        ]

        self.targets = df2.target.values
        self.centers = df2.center.values





# import torch

# class Isic2019raw(torch.utils.data.Dataset):
    
#     dataset_downloaded = False

#     def __init__(self, X_dtype=torch.float32, y_dtype=torch.int64, augmentations=None, data_path=None, vlm_prompt=True):
        
#         if not (os.path.exists(data_path)):
#             os.makedirs(data_path, exist_ok=True)

#         if not Isic2019raw.dataset_downloaded:
#             self.input_path = download_isic_dataset(data_path)
#             Isic2019raw.dataset_downloaded = True
#         else:
#             self.input_path = os.path.join(data_path, "ISIC_2019")

#         if not (os.path.exists(data_path)):
#             raise ValueError(f"The data path does not exist")
        
#         self.class_prompts = {
#                         0: "A dermoscopic image showing melanoma, a malignant tumor arising from melanocytes. It exhibits asymmetric pigmentation, irregular borders, color variegation, atypical network structures, and blue-white veils. Represents a high-grade lesion with strong metastatic potential.",
#                         1: "A dermoscopic image showing melanocytic nevus (benign mole), characterized by symmetry, regular pigment network, uniform color, and well-defined borders. Represents a benign melanocytic proliferation, often used as a baseline control class for melanoma detection.",
#                         2: "A dermoscopic image showing basal cell carcinoma, a malignant keratinocytic tumor with arborizing vessels, blue-gray ovoid nests, ulcerations, and leaf-like or spoke-wheel structures. It is locally invasive but rarely metastatic.",
#                         3: "A dermoscopic image showing actinic keratosis, a premalignant epidermal lesion caused by chronic UV exposure. Displays surface scaling, erythematous background, and strawberry pattern. May progress to squamous cell carcinoma if untreated.",
#                         4: "A dermoscopic image showing a benign keratosis-like lesion, including seborrheic keratosis, solar lentigo, or lichen-planus–like keratosis. Exhibits milia-like cysts, comedo-like openings, and sharp demarcation. Represents benign epidermal hyperplasia.",
#                         5: "A dermoscopic image showing dermatofibroma, a benign fibrohistiocytic lesion characterized by a central white scar-like area surrounded by a delicate pigment network. Lesion is firm and dome-shaped, with low malignant potential.",
#                         6: "A dermoscopic image showing a vascular lesion, such as hemangioma or angiokeratoma, marked by red, purple, or blue coloration, lacunae patterns, and absence of pigment network. Reflects benign proliferation of dermal blood vessels.",
#                         7: "A dermoscopic image showing squamous cell carcinoma, a malignant keratinocytic tumor with keratin pearls, white circles, glomerular vessels, and ulceration. Represents a more aggressive evolution from actinic keratosis, capable of metastasis."
#                     }
        
#         self.dic = {"input_preprocessed": os.path.join(self.input_path, "ISIC_2019_Training_Input_preprocessed"), "train_test_split": os.path.join(self.input_path, "train_test_split")}

#         self.X_dtype = X_dtype
#         self.y_dtype = y_dtype

#         self.vlm_prompt = vlm_prompt
        
#         df2 = pd.read_csv(self.dic["train_test_split"])

#         df2['prompts'] = df2["target"].map(self.class_prompts)
        
#         self.df = df2.copy()
        
#         images = df2.image.to_list()

#         self.image_paths = [os.path.join(self.dic["input_preprocessed"], image_name + ".jpg") for image_name in images]

#         self.targets = df2.target

#         self.augmentations = augmentations

#         self.centers = df2.center

#         self.prompts = df2.prompts

        


#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = np.array(Image.open(image_path))
#         target = self.targets[idx]

#         if self.augmentations is not None:
#             augmented = self.augmentations(image=image)
#             image = augmented["image"]

#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)

#         if self.vlm_prompt:
#             prompt = self.prompts[idx]


#         return (torch.tensor(image, dtype=self.X_dtype), torch.tensor(target, dtype=self.y_dtype), prompt)
# class Isic2019raw(torch.utils.data.Dataset):
    
#     dataset_downloaded = False

#     def __init__(self, X_dtype=torch.float32, y_dtype=torch.int64, augmentations=None, data_path=None, vlm_prompt=True, clip_tokenizer=None):
        
#         if not (os.path.exists(data_path)):
#             os.makedirs(data_path, exist_ok=True)

#         if not Isic2019raw.dataset_downloaded:
#             self.input_path = download_isic_dataset(data_path)
#             Isic2019raw.dataset_downloaded = True
#         else:
#             self.input_path = os.path.join(data_path, "ISIC_2019")

#         if not (os.path.exists(data_path)):
#             raise ValueError(f"The data path does not exist")
        
#         self.idx2label = {
#             0: "melanoma",
#             1: "melanocytic nevus",
#             2: "basal cell carcinoma",
#             3: "actinic keratosis",
#             4: "benign keratosis-like lesion",
#             5: "dermatofibroma",
#             6: "vascular lesion",
#             7: "squamous cell carcinoma",
#         }
        
#         self.dic = {"input_preprocessed": os.path.join(self.input_path, "ISIC_2019_Training_Input_preprocessed"), "train_test_split": os.path.join(self.input_path, "train_test_split")}

#         self.X_dtype = X_dtype
#         self.y_dtype = y_dtype

#         self.vlm_prompt = vlm_prompt

#         self.clip_tokenizer = clip_tokenizer
        
#         df2 = pd.read_csv(self.dic["train_test_split"])

#         prompt_template = "A clinical dermoscopy photograph of {}."

#         prompt_list = [prompt_template.format(self.idx2label[i]) for i in range(len(self.idx2label))]

#         self.text_tokens = self.clip_tokenizer(prompt_list, truncate=True)
        
        
#         self.df = df2.copy()
        
#         images = df2.image.to_list()

#         self.image_paths = [os.path.join(self.dic["input_preprocessed"], image_name + ".jpg") for image_name in images]

#         self.targets = df2.target

#         self.augmentations = augmentations

#         self.centers = df2.center


        


#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = np.array(Image.open(image_path))
#         target = self.targets[idx]

#         if self.augmentations is not None:
#             augmented = self.augmentations(image=image)
#             image = augmented["image"]

#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)

#         # if self.vlm_prompt:
#         #     prompt = self.prompts[idx]
#         if self.vlm_prompt:
#             text_enc = self.text_tokens[target]
#             return (torch.tensor(image, dtype=self.X_dtype), torch.tensor(target, dtype=self.y_dtype), text_enc)


#         return (torch.tensor(image, dtype=self.X_dtype), torch.tensor(target, dtype=self.y_dtype))



# class Fedisic2019(Isic2019raw):
#     def __init__(self,
#                  train: bool = True,
#                  X_dtype: torch.dtype = torch.float32,
#                  y_dtype: torch.dtype = torch.int64,
#                  data_path: str = None,
#                  cid: int = 0,
#                  pooled: bool = False,
#                  sz: int = 224,
#                  clip_tokenizer = None):
        
#         if train:
#             augmentations = albumentations.Compose([

#                 albumentations.RandomScale(0.07),
#                 albumentations.Rotate(50),
#                 albumentations.RandomBrightnessContrast(0.15, 0.1),
#                 albumentations.Flip(p=0.5),
#                 albumentations.Affine(shear=0.1),
#                 albumentations.RandomCrop(sz, sz),
#                 albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
#                 albumentations.Normalize(always_apply=True),
#             ]
#             )

#         else:
#             augmentations = albumentations.Compose([
#                 albumentations.CenterCrop(sz, sz),
#                 albumentations.Normalize(always_apply=True)
#             ])

#         super().__init__(X_dtype=X_dtype, y_dtype=y_dtype, augmentations=augmentations, data_path=data_path,clip_tokenizer=clip_tokenizer)

#         self.train_test = "train" if train else "test"
#         self.cid = cid
#         self.pooled = pooled

#         self.centers = [self.cid]

#         df = pd.read_csv(self.dic["train_test_split"])

#         if self.pooled:
#             df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)

#         else:
#             df2 = df[(df['fold'] == self.train_test) & (df['center'].isin(self.centers))].reset_index(drop=True)

#         images = df2.image.tolist()

#         self.image_paths = [os.path.join(self.dic["input_preprocessed"], image_name + ".jpg") for image_name in images]

#         self.targets = df2.target
#         self.centers = df2.center
        
                