import os
import ast
import glob
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class SDSDVideoDatasetRGB(Dataset):
    """RGB-only video dataset for SDSD (indoor/outdoor). Reads .npy frames (H,W,3) uint8."""

    def __init__(
        self,
        root,
        clip_len=8,
        clip_stride=2,
        base_size=256,
        crop_size=(128, 128),
        split_json=None,
        split="train",
    ):
        super().__init__()

        self.root = root
        self.clip_len = clip_len
        self.clip_stride = clip_stride

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.base_size = base_size
        self.split = split

        self.input_root = os.path.join(root, "input")
        self.gt_root = os.path.join(root, "GT")

        if split_json is not None:
            with open(split_json, "r") as f:
                split_dict = json.load(f)
            self.video_ids = sorted(split_dict[split])
        else:
            self.video_ids = sorted(os.listdir(self.gt_root))

        self.to_tensor = T.ToTensor()

        self.videos = []
        for vid in self.video_ids:
            gt_frames = sorted(glob.glob(f"{self.gt_root}/{vid}/*.npy"))
            in_frames = sorted(glob.glob(f"{self.input_root}/{vid}/*.npy"))

            if len(gt_frames) == 0 or len(in_frames) == 0:
                print(f"[SDSD Warn] Video {vid} is empty (gt={len(gt_frames)}, in={len(in_frames)}). Skipping.")
                continue

            n_frames = min(len(gt_frames), len(in_frames))
            gt_frames = gt_frames[:n_frames]
            in_frames = in_frames[:n_frames]

            self.videos.append((gt_frames, in_frames))

        self.index_list = []
        for vid_id, (gt_frames, _) in enumerate(self.videos):
            N = len(gt_frames)
            if N >= clip_len:
                for st in range(0, N - clip_len + 1, self.clip_stride):
                    self.index_list.append((vid_id, st))

        print(f"[SDSD Dataset] split={split} videos={len(self.video_ids)} clips={len(self.index_list)}")
        print(f"[SDSD Dataset] Base Resize={self.base_size} (Even Aligned) -> Crop={self.crop_size}")

    def __len__(self):
        return len(self.index_list)

    def _get_npy_size(self, path):
        """Get image size from .npy header without loading full array."""
        try:
            with open(path, 'rb') as f:
                magic = f.read(6)
                if magic[:6] == b'\x93NUMPY':
                    major = f.read(1)[0]
                    minor = f.read(1)[0]
                    if major == 1:
                        header_len = int.from_bytes(f.read(2), 'little')
                    else:
                        header_len = int.from_bytes(f.read(4), 'little')
                    header = f.read(header_len).decode('ascii')
                    header_dict = ast.literal_eval(header.strip().rstrip(','))
                    shape = header_dict['shape']
                    return shape[1], shape[0]
        except Exception:
            pass

        arr = np.load(path)
        return arr.shape[1], arr.shape[0]

    def _apply_transforms(self, tensor, crop_params, do_flip, rot_k):
        if tensor is None:
            return None
        top, left, h, w = crop_params

        if tensor.shape[-2] < h or tensor.shape[-1] < w:
            tensor = F.interpolate(tensor.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        else:
            tensor = tensor[:, top:top+h, left:left+w]

        if do_flip:
            tensor = TF.hflip(tensor)
        if rot_k > 0:
            tensor = torch.rot90(tensor, k=rot_k, dims=[1, 2])
        return tensor

    def __getitem__(self, idx):
        vid_id, st = self.index_list[idx]
        gt_frames, in_frames = self.videos[vid_id]
        is_train = (self.split == 'train')

        W_orig, H_orig = self._get_npy_size(in_frames[st])

        crop_h, crop_w = self.crop_size

        render_w, render_h = W_orig, H_orig

        if self.base_size is not None:
            scale = self.base_size / min(W_orig, H_orig)
            render_h = int(H_orig * scale)
            render_w = int(W_orig * scale)

            if render_h % 2 != 0:
                render_h += 1
            if render_w % 2 != 0:
                render_w += 1

        if render_h < crop_h or render_w < crop_w:
            scale_safe = max(crop_h / render_h, crop_w / render_w)
            render_h = int(render_h * scale_safe)
            render_w = int(render_w * scale_safe)
            if render_h % 2 != 0:
                render_h += 1
            if render_w % 2 != 0:
                render_w += 1

        if is_train:
            i = np.random.randint(0, render_h - crop_h + 1)
            j = np.random.randint(0, render_w - crop_w + 1)
            do_flip = np.random.rand() < 0.5
            rot_k = np.random.randint(0, 4)
        else:
            i = (render_h - crop_h) // 2
            j = (render_w - crop_w) // 2
            do_flip = False
            rot_k = 0

        crop_params = (i, j, crop_h, crop_w)

        rgb_in, rgb_gt = [], []

        for k in range(self.clip_len):
            idx_curr = st + k

            arr_gt = np.load(gt_frames[idx_curr])
            arr_in = np.load(in_frames[idx_curr])

            img_gt = Image.fromarray(arr_gt).resize((render_w, render_h), Image.BICUBIC)
            img_in = Image.fromarray(arr_in).resize((render_w, render_h), Image.BICUBIC)

            t_gt = self._apply_transforms(self.to_tensor(img_gt), crop_params, do_flip, rot_k)
            t_in = self._apply_transforms(self.to_tensor(img_in), crop_params, do_flip, rot_k)

            rgb_gt.append(t_gt)
            rgb_in.append(t_in)

        return (
            torch.stack(rgb_in, 0),
            torch.stack(rgb_gt, 0),
            torch.zeros(self.clip_len, 10, self.crop_size[0], self.crop_size[1]),
            torch.zeros(self.clip_len, 1, self.crop_size[0], self.crop_size[1]),
            0,
            0,
        )
