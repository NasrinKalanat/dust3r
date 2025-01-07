# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# See datasets_preprocess/preprocess_co3d.py
# --------------------------------------------------------
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

import glob
import os


class CropsDataset(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.dataset_label = 'Crops'

        self.scenes = {}
        self.camera_params = {}
        self.scene_list = []
        self.combinations = []
        self.paths = {}
        # load all scenes
        for fol in os.listdir(self.ROOT):
            splt = fol.split("_")
            cat = splt[2]
            date = splt[1][-8:]
            path = os.path.join(self.ROOT, fol)
            with open(osp.join(path, f'transforms.json'), 'r') as f:
                json_data = json.load(f)
                self.camera_params[(cat, date)] = {
                    "fl_x": json_data["fl_x"],
                    "fl_y": json_data["fl_y"],
                    "cx": json_data["cx"],
                    "cy": json_data["cy"],
                    "k1": json_data["k1"],
                    "k2": json_data["k2"],
                    "k3": json_data["k3"],
                    "p1": json_data["p1"],
                    "p2": json_data["p2"]
                }
                self.scenes[(cat, date)] = json_data["frames"]
                self.paths[(cat, date)] = path
            self.scene_list += list(self.scenes.keys())

            # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
            # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
            self.combinations += [(cat, date, i, j)
                                for i, j in itertools.combinations(range(len(json_data["frames"])), 2)
                                if 0 < abs(i - j) <= 30 and abs(i - j) % 5 == 0]

            # self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return len(self.combinations)

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.npz')

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'depths', f'frame{view_idx:06n}.jpg.geometric.png')

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
        return depthmap

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        # obj, instance = self.scene_list[idx // len(self.combinations)]
        cat, date, im1_idx, im2_idx = self.combinations[idx]
        image_pool = self.scenes[(cat, date)]

        # add a bit of randomness
        last = len(image_pool) - 1

        # if resolution not in self.invalidate[obj, instance]:  # flag invalid images
        #     self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            # if self.invalidate[obj, instance][resolution][im_idx]:
            #     # search for a valid image
            #     random_direction = 2 * rng.choice(2) - 1
            #     for offset in range(1, len(image_pool)):
            #         tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
            #         if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
            #             im_idx = tentative_im_idx
            #             break

            frame = self.scenes[(cat, date)][im_idx]

            impath = os.path.join(self.paths[(cat, date)], frame["file_path"])
            # depthpath = self._get_depthpath(obj, instance, view_idx)

            # load camera params
            camera_pose = np.array(frame["transform_matrix"], dtype=np.float32)

            # Camera intrinsics from JSON data
            intrinsics = np.array([
                [self.camera_params[(cat, date)]["fl_x"], 0, self.camera_params[(cat, date)]["cx"]],
                [0, self.camera_params[(cat, date)]["fl_y"], self.camera_params[(cat, date)]["cy"]],
                [0, 0, 1]
            ], dtype=np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)
            # depthmap = self._read_depthmap(depthpath, input_metadata)
            depthmap = rgb_image[:, :, 2]

            # if mask_bg:
            #     # load object mask
            #     maskpath = self._get_maskpath(obj, instance, frame)
            #     maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            #     maskmap = (maskmap / 255.0) > 0.1

            #     # update the depthmap with mask
            #     depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            # num_valid = (depthmap > 0.0).sum()
            # if num_valid == 0:
            #     # problem, invalidate image and retry
            #     self.invalidate[obj, instance][resolution][im_idx] = True
            #     imgs_idxs.append(im_idx)
            #     continue

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=osp.join(cat, date),
                instance=date,
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = CropsDataset(split='train', ROOT="/users/esmaeil/data_3d/data", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
