import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random


_Path2Permutes = "libdg/zdpath/jigsaw_permutes"
_FilePattern4Permutes = "permutations_{}.npy"
# transformations for each tile of jigsaw
jigsaw_tile_tr = transforms.Compose([
    transforms.RandomGrayscale(0.1),   #  FIXME: using this is cheating but seems to have a big impact on performance of jiGen
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class JigsawDsetWrapper(data.Dataset):
    """
    Convert an arbitrary torchvision dataset into image and tiles
    """

    def __init__(self, dataset,
                 jig_classes=31,   #FIXME: 31 has to be set both at algoirhtm side and scenario side now
                 tile_transformer=jigsaw_tile_tr,
                 patches=False,
                 bias_whole_image=0.7,  #FIXME: it seems bias_whole_image has an impact on performance
                 num_grids_h=3):
        """
        :param dataset:  torchvision dataset
        :param jig_classes:
        :param tile_transformer:
        :param patches:
        :param bias_whole_image:
        :param num_grids_h:
        """

        self.dataset = dataset
        self.permutations = self._retrieve_permutes_from_disk(jig_classes)
        # for 3*3 tiles, there are 9*8*7*6*5*...*1 >> 100 possibilities of permutations
        # we load from disk instead only 100 permutations
        self.num_grids_h = num_grids_h   # break the image into 3*3 tiles
        self.bias_whole_image = bias_whole_image
        if patches:   # default false
            self.patch_size = 64    # not sure what this serves for???
        self._augment_tile = tile_transformer
        if patches:    # default False, do not return patches(tiles) directly but sew them together
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):   # sew tiles together to be images
                """
                """
                return torchvision.utils.make_grid(x, nrow=self.num_grids_h, padding=0)
            self.returnFunc = make_grid

    def get_tile(self, img, n):
        """
        :param img:
        :param n:
        """
        img_size = img.shape[-1]   #  FIXME: use a better way to decide the image size
        w = float(img_size) / self.num_grids_h
        w = float(int(w)) + 1   # FIXME: extra line to ensure w=75 instead of sometimes 74 so torch.stack can fail
        # in original data, w = float(img.size[0]) / self.num_grids_h = 225/3 = 75.0 is an integer, but this can not be true for other cases
        y = int(n / self.num_grids_h)
        x = n % self.num_grids_h
        tr = transforms.ToPILImage()
        img_pil = tr(img)
        # PIL.crop((left, top, right, bottom))
        tile = img_pil.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def __getitem__(self, index):
        """__getitem__.

        :param index:
        """
        img, label = self.dataset.__getitem__(index)
        n_grids = self.num_grids_h ** 2    # divide image into num_grids_h^2 tiles
        tiles = [None] * n_grids     # list of length n_grids of image tiles
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)    # populate tile list

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        # len(self.permutations) by default is 100, so order is a random number between 0 and 101
        # order is basically the row index to choose from self.permutations which is a matrix of 100*9 usually, where 9=3*3 is the number of tiles the image is broken into
        if self.bias_whole_image:    # default is None
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:   # default
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)   # the 0th dim is the batch dimension
        # NOTE: label must be the second place so that functions like performance.get_accuracy could work!
        return self.returnFunc(data), label, int(order)     # order is the ground truth for the permutation index

    def __len__(self):
        """__len__."""
        return self.dataset.__len__()

    def _retrieve_permutes_from_disk(self, classes):
        """_retrieve_permutes_from_disk.

        :param classes:
        """
        fa = _FilePattern4Permutes.format(classes)
        path = os.path.realpath(_Path2Permutes)
        fpath = os.path.join(path, fa)
        all_perm = np.load(fpath)
        if all_perm.min() == 1:
            # from range [1,9] to [0,8]
            all_perm = all_perm - 1
        return all_perm
