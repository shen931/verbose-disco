import torch
import cv2  # 使用 OpenCV 进行图像处理
from math import sqrt, exp
from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays, get_rays_ortho

def build_laplacian_pyramid(image, num_levels):
    pyramid = []
    current_image = image
    for _ in range(num_levels):
        next_image = cv2.pyrDown(current_image)  # 下采样
        upsampled_next_image = cv2.pyrUp(next_image, dstsize=(current_image.shape[1], current_image.shape[0]))  # 上采样回原尺寸
        laplacian = current_image - upsampled_next_image  # 计算差值图像
        pyramid.append(laplacian)
        current_image = next_image
    pyramid.append(current_image)  # 最底层的高斯图像
    return pyramid
    
class ImgToPatch(object):
    def __init__(self, ray_sampler, hwf, bg_threshold=0.1, num_levels=3):
        self.ray_sampler = ray_sampler
        self.hwf = hwf  # 相机内参
        self.bg_threshold = bg_threshold  # 背景阈值
        self.num_levels = num_levels  # 拉普拉斯金字塔层数

    def __call__(self, img):
        # 构建拉普拉斯金字塔
        img_np = img.permute(0, 2, 3, 1).cpu().numpy()  # 转换为 NumPy 数组 (B, H, W, C)
        laplacian_pyramid = [build_laplacian_pyramid(img_np[i], self.num_levels) for i in range(img_np.shape[0])]
        
        rgbs = []
        for img_i in img:
            pose = torch.eye(4)  # 使用虚拟位姿来推断像素值
            _, selected_idcs, pixels_i = self.ray_sampler(H=self.hwf[0], W=self.hwf[1], focal=self.hwf[2], pose=pose)

            # 使用 OpenCV 进行前景检测（例如，通过简单的阈值分割）
            img_np = img_i.permute(1, 2, 0).cpu().numpy()  # 转换为 NumPy 数组 (H, W, C)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # 转为灰度图
            _, foreground_mask = cv2.threshold(gray, self.bg_threshold * 255, 255, cv2.THRESH_BINARY)

            # 将前景掩码转换为 Tensor
            foreground_mask_tensor = torch.tensor(foreground_mask, dtype=torch.bool)

            if selected_idcs is not None:
                fg_selected_idcs = selected_idcs[foreground_mask_tensor.view(-1)]
                if fg_selected_idcs.numel() > 0:
                    rgbs_i = img_i.flatten(1, 2).t()[fg_selected_idcs]
                else:
                    rgbs_i = img_i.flatten(1, 2).t()[selected_idcs]
            else:
                rgbs_i = torch.nn.functional.grid_sample(img_i.unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()

            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)  # (B*N)x3
        return rgbs


class RaySampler(object):
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, focal, pose):
        if self.orthographic:
            size_h, size_w = focal  # Hacky
            rays_o, rays_d = get_rays_ortho(H, W, pose, size_h, size_w)
        else:
            rays_o, rays_d = get_rays(H, W, focal, pose)

        select_inds = self.sample_rays(H, W)

        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds]
            rays_d = rays_d.view(-1, 3)[select_inds]

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds % W) / float(W) - 0.5

            hw = torch.stack([h, w]).t()

        else:
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2, 0, 1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2, 0, 1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1, 2, 0).view(-1, 3)
            rays_d = rays_d.permute(1, 2, 0).view(-1, 3)

            hw = select_inds
            select_inds = None

        return torch.stack([rays_o, rays_d]), select_inds, hw

    def sample_rays(self, H, W):
        raise NotImplementedError


class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H * W)


class AdaptiveRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1, **kwargs):
        super(AdaptiveRaySampler, self).__init__(N_samples, **kwargs)
        self.random_shift = random_shift
        self.random_scale = random_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W, foreground_mask=None):
        if foreground_mask is not None:
            # 基于前景掩码自适应调整采样
            foreground_indices = torch.nonzero(foreground_mask.view(-1)).squeeze()
            if foreground_indices.numel() > 0:
                # 从前景区域采样
                return foreground_indices[torch.randint(len(foreground_indices), (self.N_samples,))]
        # 默认返回均匀采样
        return super().sample_rays(H, W)


class FlexGridRaySampler(AdaptiveRaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1, **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, random_shift, random_scale, min_scale, max_scale, scale_anneal, **kwargs)

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1, 1, self.N_samples_sqrt),
                                         torch.linspace(-1, 1, self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # 直接返回网格以供 grid_sample 使用
        self.return_indices = False
        self.iterations = 0

    def sample_rays(self, H, W):
        if self.scale_anneal > 0:
            k_iter = self.iterations // 1000 * 3
            min_scale = max(self.min_scale, self.max_scale * exp(-k_iter * self.scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        scale = 1
        if self.random_scale:
            scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
            h = self.h * scale
            w = self.w * scale

        if self.random_shift:
            max_offset = 1 - scale.item()
            h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2
            w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2

            h += h_offset
            w += w_offset

        self.scale = scale

        return torch.cat([h, w], dim=2)