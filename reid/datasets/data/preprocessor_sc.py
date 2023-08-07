from __future__ import absolute_import

import os

from PIL import Image, ImageFilter
import cv2 
import numpy as np
from torch.utils.data import Dataset
import clip
import json



class PreProcessor(Dataset):
    def __init__(self, dataset, json_list=None, root=None, root_additional=None, transform=None, clothes_transform=None, blur_clo=False):
        super(PreProcessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.root_additional = root_additional
        self.transform = transform
        self.initialized = False
        self.clothes_transform = clothes_transform
        self.blur_clo = blur_clo
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)


    def _get_single_item(self, index):
        fname, attr_fname, pid, cid, cam = self.dataset[index]
        fpath = fname
        attr_item = 'do not change clothes'
        if int(pid)==-1:
            if self.root_additional is not None:
                fpath = os.path.join(self.root_additional, fname)
        else:
            if self.root is not None:
                fpath = os.path.join(self.root, fname)
                
        img = Image.open(fpath).convert('RGB')
        attribute = clip.tokenize(attr_item)
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, attribute, fname, attr_fname, pid, cid, cam, index