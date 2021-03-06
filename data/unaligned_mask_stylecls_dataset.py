import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_mask
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import numpy as np


class UnalignedMaskStyleClsDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        imglistA = './datasets/list/%s/%s.txt' % (opt.phase+'A', opt.dataroot)
        imglistB = './datasets/list/%s/%s.txt' % (opt.phase+'B', opt.dataroot)
        
        if not os.path.exists(imglistA) or not os.path.exists(imglistB):
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        else:
            self.A_paths = sorted(open(imglistA, 'r').read().splitlines())
            self.B_paths = sorted(open(imglistB, 'r').read().splitlines())

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print("A size:", self.A_size)
        print("B size:", self.B_size)
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        if opt.dataroot == '190613-4s':
            self.softmaxloc = os.path.join('style_features/styles2/', '1vgg19_softmax')
        elif opt.dataroot == '190613-4sn5':
            self.softmaxloc = os.path.join('style_features/styles2_sn_equal/', '1vgg19_softmax')
        elif '190613-4sn' in self.opt.dataroot:
            self.softmaxloc = os.path.join('style_features/styles2_sn/', '1vgg19_softmax')


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        basenA = os.path.basename(A_path)
        A_mask_img = Image.open(os.path.join('./datasets/list/mask/A',basenA))
        basenB = os.path.basename(B_path)
        basenB2 = basenB.replace('_fake.png','.png')
        # for added synthetic drawing
        basenB2 = basenB2.replace('_style1.png','.png')
        basenB2 = basenB2.replace('_style2.png','.png')
        basenB2 = basenB2.replace('_style1single.png','.png')
        basenB2 = basenB2.replace('_style2single.png','.png')
        B_mask_img = Image.open(os.path.join('./datasets/list/mask/B',basenB2))
        if self.opt.use_eye_mask:
            A_maske_img = Image.open(os.path.join('./datasets/list/mask/A_eyes',basenA))
            B_maske_img = Image.open(os.path.join('./datasets/list/mask/B_eyes',basenB2))
        if self.opt.use_lip_mask:
            A_maskl_img = Image.open(os.path.join('./datasets/list/mask/A_lips',basenA))
            B_maskl_img = Image.open(os.path.join('./datasets/list/mask/B_lips',basenB2))
        if self.opt.metric_inmask:
            A_maskfg_img = Image.open(os.path.join('./datasets/list/mask/A_fg',basenA))

        # apply image transformation
        transform_params_A = get_params(self.opt, A_img.size)
        transform_params_B = get_params(self.opt, B_img.size)
        A = get_transform(self.opt, transform_params_A, grayscale=(self.input_nc == 1))(A_img)
        B = get_transform(self.opt, transform_params_B, grayscale=(self.output_nc == 1))(B_img)
        A_mask = get_transform_mask(self.opt, transform_params_A, grayscale=1)(A_mask_img)
        B_mask = get_transform_mask(self.opt, transform_params_B, grayscale=1)(B_mask_img)
        if self.opt.use_eye_mask:
            A_maske = get_transform_mask(self.opt, transform_params_A, grayscale=1)(A_maske_img)
            B_maske = get_transform_mask(self.opt, transform_params_B, grayscale=1)(B_maske_img)
        if self.opt.use_lip_mask:
            A_maskl = get_transform_mask(self.opt, transform_params_A, grayscale=1)(A_maskl_img)
            B_maskl = get_transform_mask(self.opt, transform_params_B, grayscale=1)(B_maskl_img)
        if self.opt.metric_inmask:
            A_maskfg = get_transform_mask(self.opt, transform_params_A, grayscale=1)(A_maskfg_img)

        item = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask}
        if self.opt.use_eye_mask:
            item['A_maske'] = A_maske
            item['B_maske'] = B_maske
        if self.opt.use_lip_mask:
            item['A_maskl'] = A_maskl
            item['B_maskl'] = B_maskl
        if self.opt.metric_inmask:
            item['A_maskfg'] = A_maskfg
        

        softmax = np.load(os.path.join(self.softmaxloc,basenB[:-4]+'.npy'))
        softmax = torch.Tensor(softmax)
        [maxv,index] = torch.max(softmax,0)
        B_label = index
        if len(self.opt.sfeature_mode) >= 8 and self.opt.sfeature_mode[-8:] == '_softmax':
            if self.opt.one_hot:
                B_style = torch.Tensor([0.,0.,0.])
                B_style[index] = 1.
            else:
                B_style = softmax
            B_style = B_style.view(3, 1, 1)
            B_style = B_style.repeat(1, 128, 128)
            #print(index, index_B, torch.mean(B_style,(1,2)))
        elif self.opt.sfeature_mode == 'domain':
            B_style = B_label
        item['B_style'] = B_style
        item['B_label'] = B_label
        if self.opt.isTrain and self.opt.style_loss_with_weight:
            item['B_style0'] = softmax
        if self.opt.isTrain and self.opt.metricvec:
            vec = softmax
            vec = vec.view(3, 1, 1)
            vec = vec.repeat(1, 299, 299)
            item['vec'] = vec

        return item

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
