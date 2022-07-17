# coding: utf-8

# # Demo for running IFCNN to fuse multiple types of images

# Project page of IFCNN is https://github.com/uzeful/IFCNN.
# 
# If you find this code is useful for your research, please consider to cite our paper.
# ```
# @article{zhang2019IFCNN,
#   title={IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network},
#   author={Zhang, Yu and Liu, Yu and Sun, Peng and Yan, Han and Zhao, Xiaolin and Zhang, Li},
#   journal={Information Fusion},
#   volume={54},
#   pages={99--118},
#   year={2020},
#   publisher={Elsevier}
# }
# ```
# 
# Detailed procedures to use IFCNN are introduced as follows.

# ## 1. Load required libraries

# In[1]:


import os
import cv2
import time
import torch

import convert_3c
from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
from utils.myTransforms import denorm, norms, detransformcv2
import nibabel as nib

import matplotlib.pyplot as plt

# ## 2. Load the well-trained image fusion model (IFCNN-MAX)

# In[2]:


# we use fuse_scheme to choose the corresponding model, choose 0 (IFCNN-MAX) for fusing multi-focus, infrare-visual
# and multi-modal medical images, 2 (IFCNN-MEAN) for fusing multi-exposure images
fuse_scheme = 0
if fuse_scheme == 0:
    model_name = 'IFCNN-MAX'
elif fuse_scheme == 1:
    model_name = 'IFCNN-SUM'
elif fuse_scheme == 2:
    model_name = 'IFCNN-MEAN'
else:
    model_name = 'IFCNN-MAX'

# load pretrained model

model = myIFCNN(fuse_scheme=fuse_scheme)
model.load_state_dict(torch.load('snapshots/' + model_name + '.pth', map_location=torch.device('cpu')))
model.eval()

# ## 3. Use IFCNN to respectively fuse CMF, IV and MD datasets
# Fusion images are saved in the 'results' folder under your current folder.

# In[3]:


from utils.myDatasets import ImagePair
from convert_3c import ConvertTo3C

IV_filenames = ['Camp', 'Camp1', 'Dune', 'Gun', 'Navi', 'Kayak', 'Octec', 'Road', 'Road2', 'Steamboat', 'T2', 'T3',
                'Trees4906', 'Trees4917', 'Window']
MF_filenames = ['clock', 'lab', 'pepsi', 'book', 'flower', 'desk', 'seascape', 'temple', 'leopard', 'wine', 'balloon',
                'calendar', 'corner', 'craft', 'leaf', 'newspaper', 'girl', 'grass', 'toy']

datasets = ['CMF']  # Color MultiFocus, Infrared-Visual, MeDical image datasets
datasets_num = [20]  # number of image sets in each dataset
is_save = True  # if you do not want to save images, then change its value to False

mimg = nib.load('HANCT.nii').get_fdata()[:, :, :]
print(type(mimg))

begin_time = time.time()
ind = 0
for ind in range(43):

    img = nib.load('HANCT.nii').get_fdata().squeeze()[:, :, ind]
    img2 = nib.load('HANPT.nii').get_fdata().squeeze()[:, :, ind]

    c3image1 = ConvertTo3C(img).convert()
    c3image2 = ConvertTo3C(img2).convert()

    # medical image dataset: CT (MR) and MR. Number: 8
    # dataset = datasets[j]  # Medical Image
    is_gray = True  # Color (False) or Gray (True)
    mean = [0, 0, 0]  # normalization parameters
    std = [1, 1, 1]
    # load source images
    pair_loader = ImagePair(c3image1.astype(np.float32), c3image2.astype(np.float32),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                            ]))
    img10, img20 = pair_loader.get_pair()

    # image = img10
    # print("Dimension of the CT scan is:", image.shape)
    # plt.imshow(image, cmap="gray")
    # plt.show()

    img10.unsqueeze_(0)
    img20.unsqueeze_(0)
    # perform image fusion
    with torch.no_grad():
        res = model(Variable(img10), Variable(img20))
        res = denorm(mean, std, res[0]).clamp(0, 1) * 255
        res_img = res.cpu().data.numpy().astype('uint8')
        img = res_img.transpose([1, 2, 0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(type(img))
    mimg[:,:, ind] = img

bimg = nib.Nifti1Image(mimg, np.eye(4))
nib.save(bimg, 'main.nii')
# ## 4. Use IFCNN to fuse triple multi-focus images in CMF dataset
# Fusion images are saved in the 'results' folder under your current folder.

# In[4]:


from utils.myDatasets import ImageSequence

dataset = 'CMF3'  # Triple Color MultiFocus
is_save = True  # Whether to save the results
is_gray = False  # Color (False) or Gray (True)
is_folder = False  # one parameter in ImageSequence
mean = [0.485, 0.456, 0.406]  # Color (False) or Gray (True)
std = [0.229, 0.224, 0.225]

begin_time = time.time()
for ind in range(4):
    # load the sequential source images
    root = 'datasets/CMFDataset/Triple Series/'
    filename = 'lytro-{:02}'.format(ind + 1)
    paths = []
    paths.append(os.path.join('{0}-A.jpg'.format(root + filename)))
    paths.append(os.path.join('{0}-B.jpg'.format(root + filename)))
    paths.append(os.path.join('{0}-C.jpg'.format(root + filename)))
    filename = model_name + '-' + dataset + '-' + 'lytro-{:02}'.format(ind + 1)

    seq_loader = ImageSequence(is_folder, 'RGB', transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]), *paths)
    imgs = seq_loader.get_imseq()

    # perform image fusion
    with torch.no_grad():
        vimgs = []
        for idx, img in enumerate(imgs):
            img.unsqueeze_(0)
            vimgs.append(Variable(img))
        vres = model(*vimgs)
        res = denorm(mean, std, vres[0]).clamp(0, 1) * 255
        res_img = res.cpu().data.numpy().astype('uint8')
        img = Image.fromarray(res_img.transpose([1, 2, 0]))

    # save the fused image
    if is_save:
        if is_gray:
            img.convert('L').save('results/' + filename + '.png', format='PNG', compress_level=0)
            print('results/' + filename + '.png')

        else:
            img.save('results/' + filename + '.png', format='PNG', compress_level=0)
            print('results/' + filename + '.png')

# when evluating time costs, remember to stop writing images by setting is_save = False
proc_time = time.time() - begin_time
print('Total processing time of {} dataset: {:.3}s'.format(dataset, proc_time))

# # 5. Load the well-trained image fusion model (IFCNN-MEAN)

# In[5]:


# we use fuse_scheme to choose the corresponding model, 
# choose 0 (IFCNN-MAX) for fusing multi-focus, infrare-visual and multi-modal medical images, 2 (IFCNN-MEAN) for fusing multi-exposure images
fuse_scheme = 2
if fuse_scheme == 0:
    model_name = 'IFCNN-MAX'
elif fuse_scheme == 1:
    model_name = 'IFCNN-SUM'
elif fuse_scheme == 2:
    model_name = 'IFCNN-MEAN'
else:
    model_name = 'IFCNN-MAX'

# load pretrained model
model = myIFCNN(fuse_scheme=fuse_scheme)
model.load_state_dict(torch.load('snapshots/' + model_name + '.pth', map_location=torch.device('cpu')))
model.eval()
# model = model.cuda()


# ## 6. Use IFCNN to fuse various number of multi-exposure images in ME Dataset
# Fusion images are saved in the 'results' folder under your current folder.

# In[6]:


from utils.myDatasets import ImageSequence

dataset = 'ME'
is_save = True
is_gray = False
is_folder = True
toggle = True
is_save_Y = False
mean = [0, 0, 0]
std = [1, 1, 1]
begin_time = time.time()
root = 'datasets/MEDataset/'

for subdir, dirs, files in os.walk(root):
    if toggle:
        toggle = False
    else:
        # Load the sequential images in each subfolder
        paths = [subdir]
        seq_loader = ImageSequence(is_folder, 'YCbCr', transforms.Compose([
            transforms.ToTensor()]), *paths)
        imgs = seq_loader.get_imseq()

        # separate the image channels
        NUM = len(imgs)
        c, h, w = imgs[0].size()
        Cbs = torch.zeros(NUM, h, w)
        Crs = torch.zeros(NUM, h, w)
        Ys = []
        for idx, img in enumerate(imgs):
            # print(img)
            Cbs[idx, :, :] = img[1]
            Crs[idx, :, :] = img[2]
            Ys.append(img[0].unsqueeze_(0).unsqueeze_(0).repeat(1, 3, 1, 1))  # Y

        # Fuse the color channels (Cb and Cr) of the image sequence
        Cbs *= 255
        Crs *= 255
        Cb128 = abs(Cbs - 128);
        Cr128 = abs(Crs - 128);
        CbNew = sum((Cbs * Cb128) / (sum(Cb128).repeat(NUM, 1, 1)));
        CrNew = sum((Crs * Cr128) / (sum(Cr128).repeat(NUM, 1, 1)));
        CbNew[torch.isnan(CbNew)] = 128
        CrNew[torch.isnan(CrNew)] = 128

        # Fuse the Y channel of the image sequence
        imgs = norms(mean, std, *Ys)  # normalize the Y channels
        with torch.no_grad():
            vimgs = []
            for idx, img in enumerate(imgs):
                vimgs.append(Variable(img))
            vres = model(*vimgs)

        # Enhance the Y channel using CLAHE
        img = detransformcv2(vres[0], mean, std)  # denormalize the fused Y channel
        y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # generate the single y channel

        y = y / 255  # initial enhancement
        y = y * 235 + (1 - y) * 16;
        y = y.astype('uint8')

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # clahe enhancement
        cy = clahe.apply(y)

        # Merge the YCbCr channels back and covert to RGB color space
        cyrb = np.zeros([h, w, c]).astype('uint8')
        cyrb[:, :, 0] = cy
        cyrb[:, :, 1] = CrNew
        cyrb[:, :, 2] = CbNew
        rgb = cv2.cvtColor(cyrb, cv2.COLOR_YCrCb2RGB)

        # Save the fused image
        img = Image.fromarray(rgb)
        filename = subdir.split('/')[-1]
        filename = model_name + '-' + dataset + '-' + filename  # y channels are fused by IFCNN, cr and cb are weighted fused

        if is_save:
            if is_gray:
                img.convert('L').save('results/' + filename + '.png', format='PNG', compress_level=0)
                print('results/' + filename + '.png')

            else:
                img.save('results/' + filename + '.png', format='PNG', compress_level=0)
                print('results/' + filename + '.png')

# when evluating time costs, remember to stop writing images by setting is_save = False
proc_time = time.time() - begin_time
print('Total processing time of {} dataset: {:.3}s'.format(dataset, proc_time))
