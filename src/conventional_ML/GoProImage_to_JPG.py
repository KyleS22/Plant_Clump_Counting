# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:47:35 2019

@author: Hp
"""

''' Convert GoPro Images to JPG'''

from PIL import Image

train_path = os.path.join('.', 'Data/Training')
for root, dirs, files in os.walk(train_path):
    for filename in files:
        print(filename)
        im = Image.open(train_path+'//'+filename)
        rgb_im = im.convert('RGB')
        rgb_im.save(train_path+'//'+filename)