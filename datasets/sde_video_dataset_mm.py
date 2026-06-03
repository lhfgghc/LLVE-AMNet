import os
import glob
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def events_to_voxel_grid_sde(events, num_bins=10, height=260, width=346):
    """Convert SDE events (N,4: timestamp,x,y,polarity) to voxel grid at 346x260."""
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if events is None or len(events) == 0:
        return voxel

    t = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    p = events[:, 3].astype(np.float32)

    p = p * 2.0 - 1.0

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]

    if len(t) == 0:
        return voxel

    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8) * (num_bins - 1)
    bin_indices = np.clip(t_norm.astype(np.int32), 0, num_bins - 1)

    np.add.at(voxel, (bin_indices, y, x), p)
    return voxel


class SDEVideoDatasetMM(Dataset):
    """Multi-modal video dataset for SDE (EvLight release). RGB + Event.
    Directory: root/{train,test}/video_id/{low/*.png+*.npz, normal/*.png}
    Event .npz: key='arr_0', shape=(N,4), columns=[timestamp,x,y,polarity]
    """

    def __init__(
        self,
        root,
        clip_len=8,
        clip_stride=2,
        base_size=256,
        crop_size=(128, 128),
        split="train",
        split_json=None,
        use_event=True,
        num_bins=10,
        mask_mode="strict",
        p_mask_event=0.0,
        use_ir=False,
        p_mask_ir=0.0,
    ):
        super().__init__()

        self.root = root
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.base_size = base_size
        self.split = split

        self.use_event = use_event
        self.num_bins = num_bins
        self.mask_mode = mask_mode
        self.p_mask_event = p_mask_event

        split_dir = os.path.join(root, split)
        if split_json is not None and os.path.exists(split_json):
            with open(split_json, "r") as f:
                split_dict = json.load(f)
            self.video_ids = sorted(split_dict[split])
        else:
            self.video_ids = sorted(os.listdir(split_dir))

        self.to_tensor = T.ToTensor()

        self.videos = []
        for vid in self.video_ids:
            low_dir = os.path.join(split_dir, vid, "low")
            normal_dir = os.path.join(split_dir, vid, "normal")

            in_frames = sorted(glob.glob(os.path.join(low_dir, "*.png")))
            gt_frames = sorted(glob.glob(os.path.join(normal_dir, "*.png")))

            if len(gt_frames) == 0 or len(in_frames) == 0:
                print(f"[Warn] Video {vid}: low={len(in_frames)} gt={len(gt_frames)}, skipping.")
                continue

            evt_frames = None
            if self.use_event:
                all_npz = sorted(glob.glob(os.path.join(low_dir, "*.npz")))
                evt_files = [f for f in all_npz if "lowlight_event" not in os.path.basename(f)]
                if len(evt_files) > 0:
                    evt_frames = evt_files

            n = min(len(in_frames), len(gt_frames))
            self.videos.append((gt_frames[:n], in_frames[:n], evt_frames))

        self.index_list = []
        for vid_id, (gt_frames, _, _) in enumerate(self.videos):
            N = len(gt_frames)
            if N >= clip_len:
                for st in range(0, N - clip_len + 1, self.clip_stride):
                    self.index_list.append((vid_id, st))

        evt_count = sum(1 for v in self.videos if v[2] is not None)
        print(f"[SDE-MM Dataset] split={split} videos={len(self.video_ids)} clips={len(self.index_list)}")
        print(f"[SDE-MM Dataset] Base Resize={self.base_size} -> Crop={self.crop_size}")
        print(f"[SDE-MM Dataset] use_event={self.use_event} | videos_with_event={evt_count}")

    def __len__(self):
        return len(self.index_list)

    def _load_event_voxel(self, path, render_h, render_w):
        """Load event .npz and convert to resized voxel grid."""
        try:
            with np.load(path) as f:
                arr = f['arr_0']
                events = arr.astype(np.float64)

            voxel = events_to_voxel_grid_sde(
                events,
                num_bins=self.num_bins,
                height=260,
                width=346,
            )
            voxel = torch.from_numpy(voxel)

            if voxel.shape[-2] != render_h or voxel.shape[-1] != render_w:
                voxel = F.interpolate(
                    voxel.unsqueeze(0),
                    size=(render_h, render_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

            return voxel
        except Exception as e:
            print(f"[Error] Failed to load event {path}: {e}")
            return None

    def _find_event_for_frame(self, evt_frames, frame_path):
        """Find matching event .npz by timestamp filename."""
        if evt_frames is None:
            return None
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        evt_dir = os.path.dirname(evt_frames[0])
        evt_path = os.path.join(evt_dir, stem + ".npz")
        if os.path.exists(evt_path):
            return evt_path
        return None

    def _apply_transforms(self, tensor, crop_params, do_flip, rot_k):
        if tensor is None:
            return None
        top, left, h, w = crop_params
        if tensor.shape[-2] < h or tensor.shape[-1] < w:
            tensor = F.interpolate(tensor.unsqueeze(0), size=(h, w),
                                   mode='bilinear', align_corners=False).squeeze(0)
        else:
            tensor = tensor[:, top:top + h, left:left + w]
        if do_flip:
            tensor = TF.hflip(tensor)
        if rot_k > 0:
            tensor = torch.rot90(tensor, k=rot_k, dims=[1, 2])
        return tensor

    def __getitem__(self, idx):
        vid_id, st = self.index_list[idx]
        gt_frames, in_frames, evt_frames = self.videos[vid_id]
        is_train = (self.split == 'train')

        img_tmp = Image.open(in_frames[st])
        W_orig, H_orig = img_tmp.size
        img_tmp.close()

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
        evt_clip = []

        for k in range(self.clip_len):
            idx_curr = st + k

            img_gt = Image.open(gt_frames[idx_curr]).convert("RGB").resize(
                (render_w, render_h), Image.BICUBIC)
            img_in = Image.open(in_frames[idx_curr]).convert("RGB").resize(
                (render_w, render_h), Image.BICUBIC)

            t_gt = self._apply_transforms(self.to_tensor(img_gt), crop_params, do_flip, rot_k)
            t_in = self._apply_transforms(self.to_tensor(img_in), crop_params, do_flip, rot_k)

            rgb_gt.append(t_gt)
            rgb_in.append(t_in)

            if self.use_event and evt_frames is not None:
                evt_path = self._find_event_for_frame(evt_frames, in_frames[idx_curr])
                if evt_path is not None:
                    t_evt = self._load_event_voxel(evt_path, render_h, render_w)
                    if t_evt is not None:
                        t_evt = self._apply_transforms(t_evt, crop_params, do_flip, rot_k)
                        evt_clip.append(t_evt)
                    else:
                        evt_clip.append(None)
                else:
                    evt_clip.append(None)
            else:
                evt_clip.append(None)

        evt_exists = not all(e is None for e in evt_clip)
        evt_final = torch.zeros(self.clip_len, self.num_bins, crop_h, crop_w)
        s_evt = 0

        if self.mask_mode == "strict":
            if evt_exists:
                none_count = sum(1 for e in evt_clip if e is None)
                if none_count == 0:
                    evt_final = torch.stack(evt_clip, 0)
                    s_evt = 1
        elif self.mask_mode == "dropout":
            if evt_exists and (np.random.rand() >= self.p_mask_event):
                try:
                    zeros = torch.zeros((self.num_bins, crop_h, crop_w))
                    evt_list = [e if e is not None else zeros for e in evt_clip]
                    evt_final = torch.stack(evt_list, 0)
                    s_evt = 1
                except:
                    s_evt = 0

        ir_final = torch.full((self.clip_len, 1, crop_h, crop_w), -1.0)
        s_ir = 0

        return (
            torch.stack(rgb_in, 0),
            torch.stack(rgb_gt, 0),
            evt_final,
            ir_final,
            s_evt,
            s_ir,
        )
