import json
import os
import pickle

import imageio
import numpy as np
import PIL.Image
import skimage.morphology
import torch
import spconv.pytorch as spconv
import torchvision

from former3d import utils


img_mean_rgb = np.array([127.71, 114.66, 99.32], dtype=np.float32)
img_std_rgb = np.array([75.31, 73.53, 71.66], dtype=np.float32)


def load_tsdf(tsdf_dir, scene_name):
    tsdf_fname = os.path.join(tsdf_dir, scene_name, "full_tsdf_layer0.npz")
    with np.load(tsdf_fname) as tsdf_04_npz:
        tsdf = tsdf_04_npz["arr_0"]

    pkl_fname = os.path.join(tsdf_dir, scene_name, "tsdf_info.pkl")
    with open(pkl_fname, "rb") as tsdf_pkl:
        tsdf_info = pickle.load(tsdf_pkl)
        origin = tsdf_info['vol_origin']
        voxel_size = tsdf_info['voxel_size']

    return tsdf, origin, voxel_size


def reflect_pose(pose, plane_pt=None, plane_normal=None):
    pts = pose @ np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    plane_pt = np.array([*plane_pt, 1], dtype=np.float32)

    pts = pts - plane_pt[None, :, None]

    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    m = np.zeros((4, 4), dtype=np.float32)
    m[:3, :3] = np.eye(3) - 2 * plane_normal[None].T @ plane_normal[None]

    pts = m @ pts + plane_pt[None, :, None]

    result = np.eye(4, dtype=np.float32)[None].repeat(len(pose), axis=0)
    result[:, :, :3] = pts[:, :, :3] - pts[:, :, 3:]
    result[:, :, 3] = pts[:, :, 3]
    return result


def get_proj_mats(intr, pose, factors):
    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    k[0] = k[0] * factors[0]
    k[1] = k[1] * factors[0]
    proj_lowres = k @ pose

    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    k[0] = k[0] * factors[1]
    k[1] = k[1] * factors[1]
    proj_midres = k @ pose

    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    k[0] = k[0] * factors[2]
    k[1] = k[1] * factors[2]
    proj_highres = k @ pose

    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    proj_depth = k @ pose

    return {
        "coarse": proj_lowres,
        "medium": proj_midres,
        "fine": proj_highres,
        "fullres": proj_depth,
    }


def load_rgb_imgs(rgb_imgfiles, imheight, imwidth, augment=False):
    if augment:
        transforms = [
            (
                torchvision.transforms.functional.adjust_brightness,
                np.random.uniform(0.5, 1.5),
            ),
            (
                torchvision.transforms.functional.adjust_contrast,
                np.random.uniform(0.5, 1.5),
            ),
            (
                torchvision.transforms.functional.adjust_hue,
                np.random.uniform(-0.05, 0.05),
            ),
            (
                torchvision.transforms.functional.adjust_saturation,
                np.random.uniform(0.5, 1.5),
            ),
            (
                torchvision.transforms.functional.gaussian_blur,
                7,
                np.random.randint(1, 4),
            ),
        ]
        transforms = [
            transforms[i]
            for i in np.random.choice(len(transforms), size=2, replace=False)
        ]

    rgb_imgs = np.empty((len(rgb_imgfiles), imheight, imwidth, 3), dtype=np.float32)
    for i, f in enumerate(rgb_imgfiles):
        img = PIL.Image.open(f)
        if augment:
            for t, *params in transforms:
                img = t(img, *params)
        rgb_imgs[i] = img

    rgb_imgs -= img_mean_rgb
    rgb_imgs /= img_std_rgb
    rgb_imgs = np.transpose(rgb_imgs, (0, 3, 1, 2))
    return rgb_imgs


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        info_files,
        tsdf_dir,
        n_imgs,
        cropsize,
        voxel_size=0.04,
        augment=True,
        load_extra=False,
    ):
        self.info_files = info_files
        self.n_imgs = n_imgs
        self.cropsize = np.array(cropsize)
        self.voxel_size = voxel_size
        self.augment = augment
        self.load_extra = load_extra
        self.tsdf_dir = tsdf_dir

        self.tmin = 0.1
        self.rmin_deg = 15

    def __len__(self):
        return len(self.info_files)

    def getitem(self, ind, **kwargs):
        return self.__getitem__(ind, **kwargs)

    def __getitem__(self, ind):
        with open(self.info_files[ind], "r") as f:
            info = json.load(f)

        scene_name = info["scene"]
        tsdf_04, origin, _ = load_tsdf(self.tsdf_dir, scene_name)

        rgb_imgfiles = [frame["filename_color"] for frame in info["frames"]]
        depth_imgfiles = [frame["filename_depth"] for frame in info["frames"]]
        pose = np.empty((len(info["frames"]), 4, 4), dtype=np.float32)
        for i, frame in enumerate(info["frames"]):
            pose[i] = frame["pose"]
        intr = np.array(info["intrinsics"], dtype=np.float32)

        test_img = imageio.imread(rgb_imgfiles[0])
        imheight, imwidth, _ = test_img.shape

        assert not np.any(np.isinf(pose) | np.isnan(pose))

        seen_coords = np.argwhere(np.abs(tsdf_04) < 0.999) * self.voxel_size + origin
        i = np.random.randint(len(seen_coords))
        anchor_pt = seen_coords[i]
        offset = np.array(
            [
                np.random.uniform(self.voxel_size, self.cropsize[0] * self.voxel_size - self.voxel_size),
                np.random.uniform(self.voxel_size, self.cropsize[1] * self.voxel_size - self.voxel_size),
                np.random.uniform(self.voxel_size, self.cropsize[2] * self.voxel_size - self.voxel_size),
            ]
        )
        minbound = anchor_pt - offset
        maxbound = minbound + self.cropsize.astype(np.float32) * self.voxel_size

        # the GT TSDF will be sampled at these points
        x = np.arange(minbound[0], maxbound[0], self.voxel_size, dtype=np.float32)
        y = np.arange(minbound[1], maxbound[1], self.voxel_size, dtype=np.float32)
        z = np.arange(minbound[2], maxbound[2], self.voxel_size, dtype=np.float32)
        x = x[: self.cropsize[0]]
        y = y[: self.cropsize[0]]
        z = z[: self.cropsize[0]]
        yy, xx, zz = np.meshgrid(y, x, z)
        sample_pts = np.stack([xx, yy, zz], axis=-1)

        flip = False
        if self.augment:
            center = np.zeros((4, 4), dtype=np.float32)
            center[:3, 3] = anchor_pt

            # rotate
            t = np.random.uniform(0, 2 * np.pi)
            R = np.array(
                [
                    [np.cos(t), -np.sin(t), 0, 0],
                    [np.sin(t), np.cos(t), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )

            shape = sample_pts.shape
            sample_pts = (
                R[:3, :3] @ (sample_pts.reshape(-1, 3) - center[:3, 3]).T
            ).T + center[:3, 3]
            sample_pts = sample_pts.reshape(shape)

            # flip
            if np.random.uniform() > 0.5:
                flip = True
                sample_pts[..., 0] = -(sample_pts[..., 0] - center[0, 3]) + center[0, 3]

        selected_frame_inds = np.array(
            utils.remove_redundant(pose, self.rmin_deg, self.tmin)
        )
        if self.n_imgs is not None:
            if len(selected_frame_inds) < self.n_imgs:
                # after redundant frame removal we can end up with too few frames--
                # add some back in
                avail_inds = list(set(np.arange(len(pose))) - set(selected_frame_inds))
                n_needed = self.n_imgs - len(selected_frame_inds)
                extra_inds = np.random.choice(avail_inds, size=n_needed, replace=False)
                selected_frame_inds = np.concatenate((selected_frame_inds, extra_inds))
            elif len(selected_frame_inds) == self.n_imgs:
                ...
            else:
                # after redundant frame removal we still have more than the target # images--
                # remove even more
                pose = pose[selected_frame_inds]
                rgb_imgfiles = [rgb_imgfiles[i] for i in selected_frame_inds]
                depth_imgfiles = [depth_imgfiles[i] for i in selected_frame_inds]

                selected_frame_inds, score = utils.frame_selection(
                    pose,
                    intr,
                    imwidth,
                    imheight,
                    sample_pts.reshape(-1, 3)[::100],  # every 100th pt for efficiency
                    self.tmin,
                    self.rmin_deg,
                    self.n_imgs,
                )
        pose = pose[selected_frame_inds]
        rgb_imgfiles = [rgb_imgfiles[i] for i in selected_frame_inds]
        depth_imgfiles = [depth_imgfiles[i] for i in selected_frame_inds]

        if self.augment:
            pose = np.linalg.inv(R) @ (pose - center) + center
            if flip:
                pose = reflect_pose(
                    pose,
                    plane_pt=center[:3, 3],
                    plane_normal=-np.array(
                        [np.cos(-t), np.sin(-t), 0], dtype=np.float32
                    ),
                )
        pose[:, :3, 3] -= minbound

        grid = (sample_pts - origin) / (
            (np.array(tsdf_04.shape, dtype=np.float32) - 1) * self.voxel_size
        ) * 2 - 1
        grid = grid[..., [2, 1, 0]]

        tsdf_04_n = torch.nn.functional.grid_sample(
            torch.from_numpy(tsdf_04)[None, None],
            torch.from_numpy(grid[None]),
            align_corners=False,
            mode="nearest",
        )[0, 0].numpy()

        tsdf_04_b = torch.nn.functional.grid_sample(
            torch.from_numpy(tsdf_04)[None, None],
            torch.from_numpy(grid[None]),
            align_corners=False,
            mode="bilinear",
        )[0, 0].numpy()

        tsdf_04 = tsdf_04_b
        inds = np.abs(tsdf_04_n) > 0.999
        tsdf_04[inds] = tsdf_04_n[inds]
        oob_inds = np.any(np.abs(grid) >= 1, axis=-1)
        tsdf_04[oob_inds] = 1

        occ_04 = np.abs(tsdf_04) < 0.999
        seen_04 = tsdf_04 < 0.999

        # seems like a bug -- dilation should happen before cropping
        occ_08 = skimage.morphology.dilation(occ_04, selem=np.ones((3, 3, 3)))
        not_occ_08 = seen_04 & ~occ_08
        occ_08 = occ_08[::2, ::2, ::2]
        not_occ_08 = not_occ_08[::2, ::2, ::2]
        seen_08 = occ_08 | not_occ_08

        occ_16 = skimage.morphology.dilation(occ_08, selem=np.ones((3, 3, 3)))
        not_occ_16 = seen_08 & ~occ_16
        occ_16 = occ_16[::2, ::2, ::2]
        not_occ_16 = not_occ_16[::2, ::2, ::2]
        seen_16 = occ_16 | not_occ_16

        spatial_shape_04 = occ_04.shape
        spatial_shape_08 = occ_08.shape
        spatial_shape_16 = occ_16.shape

        rgb_imgs = load_rgb_imgs(rgb_imgfiles, imheight, imwidth, augment=self.augment)

        depth_imgs = np.empty((len(depth_imgfiles), imheight, imwidth), dtype=np.uint16)
        for i, f in enumerate(depth_imgfiles):
            depth_imgs[i] = imageio.imread(f)
        depth_imgs = depth_imgs / np.float32(1000)

        if self.augment and flip:
            # flip images
            depth_imgs = np.ascontiguousarray(np.flip(depth_imgs, axis=-1))
            rgb_imgs = np.ascontiguousarray(np.flip(rgb_imgs, axis=-1))
            intr[0, 0] *= -1

        inds_04 = np.argwhere(
            (tsdf_04 < 0.999) | np.all(tsdf_04 > 0.999, axis=-1, keepdims=True)
        )
        inds_08 = np.argwhere(seen_08 | np.all(~seen_08, axis=-1, keepdims=True))
        inds_16 = np.argwhere(seen_16 | np.all(~seen_16, axis=-1, keepdims=True))

        tsdf_04 = tsdf_04[inds_04[:, 0], inds_04[:, 1], inds_04[:, 2]]
        occ_08 = occ_08[inds_08[:, 0], inds_08[:, 1], inds_08[:, 2]].astype(np.float32)
        occ_16 = occ_16[inds_16[:, 0], inds_16[:, 1], inds_16[:, 2]].astype(np.float32)

        batch_size = 1
        tsdf_04 = spconv.SparseConvTensor(
            torch.from_numpy(tsdf_04[:, None]), torch.cat([torch.zeros(inds_04.shape[0], 1), torch.from_numpy(inds_04)], dim=1).int(), 
            spatial_shape_04, batch_size)
        occ_08 = spconv.SparseConvTensor(
            torch.from_numpy(occ_08[:, None]), torch.cat([torch.zeros(inds_08.shape[0], 1), torch.from_numpy(inds_08)], dim=1).int(), 
            spatial_shape_08, batch_size)
        occ_16 = spconv.SparseConvTensor(
            torch.from_numpy(occ_16[:, None]), torch.cat([torch.zeros(inds_16.shape[0], 1), torch.from_numpy(inds_16)], dim=1).int(), 
            spatial_shape_16, batch_size)

        cam_positions = pose[:, :3, 3]

        # world to camera
        pose_w2c = np.linalg.inv(pose)

        # refers to the downsampling ratios at various levels of the CNN feature maps
        factors = np.array([1 / 16, 1 / 8, 1 / 4])
        proj_mats = get_proj_mats(intr, pose_w2c, factors)

        # generate dense initial grid
        x = torch.arange(seen_16.shape[0], dtype=torch.int32)
        y = torch.arange(seen_16.shape[1], dtype=torch.int32)
        z = torch.arange(seen_16.shape[2], dtype=torch.int32)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        input_voxels_16 = torch.stack(
            (xx.flatten(), yy.flatten(), zz.flatten()), dim=-1
        )

        # the scene has been adjusted to origin 0
        origin = np.zeros(3, dtype=np.float32)

        scene = {
            "input_voxels_16": input_voxels_16,
            "rgb_imgs": rgb_imgs,
            "cam_positions": cam_positions,
            "proj_mats": proj_mats,
            "voxel_gt_fine": tsdf_04,
            "voxel_gt_medium": occ_08,
            "voxel_gt_coarse": occ_16,
            "scene_name": scene_name,
            "index": ind,
            "depth_imgs": depth_imgs,
            "origin": origin,
        }

        if self.load_extra:
            scene.update(
                {
                    "intr_fullres": intr,
                    "pose": pose_w2c,
                }
            )
        return scene


if __name__ == "__main__":

    import glob
    import yaml

    import matplotlib.pyplot as plt
    import open3d as o3d
    import skimage.measure

    import collate

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    with open(os.path.join(config["scannet_dir"], "scannetv2_train.txt"), "r") as f:
        train_split = f.read().split()

    with open(os.path.join(config["scannet_dir"], "scannetv2_val.txt"), "r") as f:
        val_split = f.read().split()

    info_files = sorted(
        glob.glob(os.path.join(config["scannet_dir"], "scans/*/info.json"))
    )
    train_info_files = [
        f for f in info_files if os.path.basename(os.path.dirname(f)) in train_split
    ]
    val_info_files = [
        f for f in info_files if os.path.basename(os.path.dirname(f)) in val_split
    ]

    dset = Dataset(
        train_info_files,
        config["tsdf_dir"],
        35,
        (48, 48, 32),
        augment=True,
        load_extra=True,
    )

    loader = torch.utils.data.DataLoader(
        dset, batch_size=2, collate_fn=collate.sparse_collate_fn
    )
    batch = next(iter(loader))
    resolutions = {
        "coarse": 0.16,
        "medium": 0.08,
        "fine": 0.04,
    }

    # get voxel gt pcds
    batch_ind = 0
    voxel_gt_pcds = []
    for resname, res in resolutions.items():
        voxel_gt = batch[f"voxel_gt_{resname}"]
        batch_mask = voxel_gt.C[:, 3] == batch_ind
        coords = voxel_gt.C[batch_mask, :3] * res + batch["origin"][0]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coords.numpy()))

        vals = voxel_gt.F[batch_mask].float()
        vals = vals - vals.min()
        vals = vals / vals.max()
        pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(vals)[:, :3])
        voxel_gt_pcds.append(pcd)

    # get depth pcd
    depth_imgs = batch["depth_imgs"][batch_ind]
    imheight = depth_imgs.shape[1]
    imwidth = depth_imgs.shape[2]
    u = np.arange(imwidth)
    v = np.arange(imheight)
    uu, vv = np.meshgrid(u, v)
    uv = np.c_[uu.flatten(), vv.flatten(), np.ones_like(uu.flatten())]
    k = batch["intr_fullres"][batch_ind]
    pix_vecs = uv @ np.linalg.inv(k.T)
    depth_pts = []
    for i in range(len(depth_imgs)):
        pose = batch["pose"][batch_ind][i].numpy()
        depth = depth_imgs[i].flatten().numpy()
        valid = depth > 0
        xyz_cam = pix_vecs[valid] * depth[valid, None]
        xyz_world = (
            np.c_[xyz_cam, np.ones((len(xyz_cam), 1))] @ np.linalg.inv(pose).T
        )[:, :3]
        depth_pts.append(xyz_world)
    depth_pts = np.concatenate(depth_pts, axis=0)
    depth_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_pts))

    # get gt high res surface
    batch_mask = batch["voxel_gt_fine"].C[:, 3] == batch_ind
    coords = batch["voxel_gt_fine"].C[batch_mask, :3].numpy()
    tsdf = batch["voxel_gt_fine"].F[batch_mask].numpy()
    tsdf = utils.to_vol(coords, tsdf)
    mesh = utils.to_mesh(
        -tsdf,
        level=0,
        mask=~np.isnan(tsdf),
        origin=batch["origin"][batch_ind].numpy(),
        voxel_size=0.04,
    )
    mesh.compute_vertex_normals()

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    utils.visualize([*voxel_gt_pcds, depth_pcd, mesh, axes])
