import glob
import torch
import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset

from pdb import set_trace as stx

'''
    将整个像素坐标空间对应的像素值即 projection: [b, 256, 256] 划分成 [bx8x8, 32, 32]
'''
# jzd add 2024.10.17
def proj_window_partition(x, window_size):
    """
    x: [256, 256]
    return out: [8*8, 32, 32], where n = window_size[0]*window_size[1] is the length of sentence
    然后 n, c 内部计算 self-attention ?
    """
    # stx()
    h,w = x.shape       # x.shape = [128, 128], window_size = (16, 16)
    x = x.view(h // window_size[0], window_size[0], w // window_size[1], window_size[1]) # [128, 128] -> [8, 16, 8, 16]
    windows = x.permute(0, 2, 1, 3).contiguous().view(-1, window_size[0], window_size[1]) # [8, 32, 8, 32] -> [8, 8, 32, 32] -> [64, 32, 32]
    
    return windows

class ImageDataset(VisionDataset):
    """
    Load images from multiple data directories.
    Folder structure: data_dir/filename.png
    """

    # jzd add 2024.10.17
    # def __init__(self, data_dirs, transforms=None):
    def __init__(self, data_dirs, window_size = [16, 16], window_num = 4, transforms=None):
        # Use multiple root folders
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]

        # initialize base class
        VisionDataset.__init__(self, root=data_dirs, transform=transforms)

        self.filenames = []
        self.window_size = window_size              # jzd add 2024.10.17
        self.window_num = window_num                 # jzd add 2024.10.17
        root = []

        for ddir in self.root:
            filenames = self._get_files(ddir)
            self.filenames.extend(filenames)
            root.append(ddir)

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _get_files(root_dir):
        return glob.glob(f'{root_dir}/*.png') + glob.glob(f'{root_dir}/*.jpg')

#TODO 这里需要修改采样方法为mlg
    def __getitem__(self, idx):
        filename = self.filenames[idx]         #data\knee_xrays_360\05_xray0060.png
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)        #torch.Size([3, 128, 128])    
        return img
    
    # stx()

    # jzd add 2024.10.17
    # def __getitem__(self, idx):
    #     filename = self.filenames[idx]
    #     # img = Image.open(filename).convert('RGB')
    #     img = Image.open(filename)
        
    #     if self.transform is not None:
    #         img = self.transform(img)         #torch.Size([1, 128, 128]) 
    #         img = img.squeeze(0)            #torch.Size([128, 128])

    #         img_window = proj_window_partition(img, self.window_size)  # [128, 128] -> [64, 16, 16]
    #         # 全是 valid 的 window
    #         img_window_valid_indx = ((img_window > 0).sum(dim=-1).sum(dim=-1) == self.window_size[0] * self.window_size[1])
    #         # print(img_window_valid_indx.shape)    #torch.Size([64])
    #         # 选取 window_inds
    #         select_inds_window = np.random.choice(img_window_valid_indx.shape[0], size=[self.window_num], replace=False) # 从 0 ~ 64-1 中选取 window_num 个值
        
    #         img_window_select = img_window[select_inds_window]      # [4, 16, 16]
    #         # print(img_window_select.shape)
    #         selected_img_window = img_window_select.flatten()     #torch.Size([1024])
    #         # print(selected_img_window.shape)
        
    #         total_inds = [i for i in range(img_window.shape[0])]
    #         # print(total_inds)        #[0, 1, 2, ..., 63]
    #         else_inds = [x for x in total_inds if x not in select_inds_window]
    #         # print(else_inds)
    #         img_window_else = img_window[else_inds]         #torch.Size([60, 16, 16])
    #         # print(img_window_else.shape)

    #         else_inds_pixel_valid = img_window_else > 0        #torch.Size([60, 16, 16])
    #         # print(else_inds_pixel_valid.shape)           

    #         img_else_valid = img_window_else[else_inds_pixel_valid]
    #         # print(img_else_valid.shape)          #torch.Size([4726])

    #         else_valid_select_index = np.random.choice(img_else_valid.shape[0], replace=False)
    #         print(else_valid_select_index)           #451

    #         selected_img_else = img_else_valid[else_valid_select_index]
                    
    #         # print(selected_img_window.shape)        #torch.Size([1024])
    #         # print(selected_img_else.shape)            #torch.Size([])
    #         selected_projs_window_valid = torch.concat([selected_img_window, selected_img_else], dim=0)     # [num]
        

    #     return selected_projs_window_valid


class Carla(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(Carla, self).__init__(*args, **kwargs)

# jzd add 2024.10.15
class Knee(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(Knee, self).__init__(*args, **kwargs)
# jzd add 2024.10.30
class DRR(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(DRR, self).__init__(*args, **kwargs)


class CelebA(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CelebA, self).__init__(*args, **kwargs)


class CUB(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CUB, self).__init__(*args, **kwargs)
        

class Cats(ImageDataset):
    def __init__(self, *args, **kwargs):
      super(Cats, self).__init__(*args, **kwargs)
    
    @staticmethod
    def _get_files(root_dir):
      return glob.glob(f'{root_dir}/CAT_*/*.jpg')


class CelebAHQ(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(CelebAHQ, self).__init__(*args, **kwargs)
    
    def _get_files(self, root):
        return glob.glob(f'{root}/*.npy')
    
    def __getitem__(self, idx):
        img = np.load(self.filenames[idx]).squeeze(0).transpose(1,2,0)
        if img.dtype == np.uint8:
            pass
        elif img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        else:
            raise NotImplementedError
        img = Image.fromarray(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img
