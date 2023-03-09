import argparse
import json
import os
import yaml
import imageio
import numpy as np
import open3d as o3d
import tqdm

import torch
import pytorch_lightning as pl
import spconv.pytorch as spconv

from former3d import data, lightningmodel, utils
from former3d.net3d.sparse3d import bxyz2xyzb


def load_model(ckpt_file, use_proj_occ, config):
    model = lightningmodel.LightningModel.load_from_checkpoint(
        ckpt_file,
        config=config,
    )
    model.sdfformer.use_proj_occ = use_proj_occ
    model = model.cuda()
    model = model.eval()
    model.requires_grad_(False)
    return model


def load_scene(info_file):
    with open(info_file, "r") as f:
        info = json.load(f)

    rgb_imgfiles = [frame["filename_color"] for frame in info["frames"]]
    depth_imgfiles = [frame["filename_depth"] for frame in info["frames"]]
    pose = np.empty((len(info["frames"]), 4, 4), dtype=np.float32)
    for i, frame in enumerate(info["frames"]):
        pose[i] = frame["pose"]
    intr = np.array(info["intrinsics"], dtype=np.float32)
    return rgb_imgfiles, depth_imgfiles, pose, intr


def get_scene_bounds(pose, intr, imheight, imwidth, frustum_depth):
    frust_pts_img = np.array(
        [
            [0, 0],
            [imwidth, 0],
            [imwidth, imheight],
            [0, imheight],
        ]
    )
    frust_pts_cam = (
        np.linalg.inv(intr) @ np.c_[frust_pts_img, np.ones(len(frust_pts_img))].T
    ).T * frustum_depth
    frust_pts_world = (
        pose @ np.c_[frust_pts_cam, np.ones(len(frust_pts_cam))].T
    ).transpose(0, 2, 1)[..., :3]

    minbound = np.min(frust_pts_world, axis=(0, 1))
    maxbound = np.max(frust_pts_world, axis=(0, 1))
    return minbound, maxbound


def get_tiles(minbound, maxbound, cropsize_voxels_fine, voxel_size_fine):
    cropsize_m = cropsize_voxels_fine * voxel_size_fine

    assert np.all(cropsize_voxels_fine % 4 == 0)
    cropsize_voxels_coarse = cropsize_voxels_fine // 4
    voxel_size_coarse = voxel_size_fine * 4

    ncrops = np.ceil((maxbound - minbound) / cropsize_m).astype(int)
    x = np.arange(ncrops[0], dtype=np.int32) * cropsize_voxels_coarse[0]
    y = np.arange(ncrops[1], dtype=np.int32) * cropsize_voxels_coarse[1]
    z = np.arange(ncrops[2], dtype=np.int32) * cropsize_voxels_coarse[2]
    yy, xx, zz = np.meshgrid(y, x, z)
    tile_origin_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    x = np.arange(0, cropsize_voxels_coarse[0], dtype=np.int32)
    y = np.arange(0, cropsize_voxels_coarse[1], dtype=np.int32)
    z = np.arange(0, cropsize_voxels_coarse[2], dtype=np.int32)
    yy, xx, zz = np.meshgrid(y, x, z)
    base_voxel_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    tiles = []
    for origin_ind in tile_origin_inds:
        origin = origin_ind * voxel_size_coarse + minbound
        tile = {
            "origin_ind": origin_ind,
            "origin": origin.astype(np.float32),
            "maxbound_ind": origin_ind + cropsize_voxels_coarse,
            "voxel_inds": torch.from_numpy(base_voxel_inds + origin_ind),
            "voxel_coords": torch.from_numpy(
                base_voxel_inds * voxel_size_coarse + origin
            ).float(),
            "voxel_features": torch.empty(
                (len(base_voxel_inds), 0), dtype=torch.float32
            ),
            "voxel_logits": torch.empty((len(base_voxel_inds), 0), dtype=torch.float32),
        }
        tiles.append(tile)
    return tiles


def frame_selection(tiles, pose, intr, imheight, imwidth, n_imgs, rmin_deg, tmin):
    sparsified_frame_inds = np.array(utils.remove_redundant(pose, rmin_deg, tmin))

    if len(sparsified_frame_inds) < n_imgs:
        # after redundant frame removal we can end up with too few frames--
        # add some back in
        avail_inds = list(set(np.arange(len(pose))) - set(sparsified_frame_inds))
        n_needed = n_imgs - len(sparsified_frame_inds)
        extra_inds = np.random.choice(avail_inds, size=n_needed, replace=False)
        selected_frame_inds = np.concatenate((sparsified_frame_inds, extra_inds))
    else:
        selected_frame_inds = sparsified_frame_inds

    for i, tile in enumerate(tiles):
        if len(selected_frame_inds) > n_imgs:
            sample_pts = tile["voxel_coords"].numpy()
            cur_frame_inds, score = utils.frame_selection(
                pose[selected_frame_inds],
                intr,
                imwidth,
                imheight,
                sample_pts,
                tmin,
                rmin_deg,
                n_imgs,
            )
            tile["frame_inds"] = selected_frame_inds[cur_frame_inds]
        else:
            tile["frame_inds"] = selected_frame_inds
    return tiles


def get_img_feats(sdfformer, imheight, imwidth, proj_mats, rgb_imgfiles, cam_positions):
    imsize = np.array([imheight, imwidth])
    dims = {
        "coarse": imsize // 16,
        "medium": imsize // 8,
        "fine": imsize // 4,
    }
    feats_2d = {
        "coarse": torch.empty(
            (1, len(rgb_imgfiles), 80, *dims["coarse"]), dtype=torch.float16
        ),
        "medium": torch.empty(
            (1, len(rgb_imgfiles), 40, *dims["medium"]), dtype=torch.float16
        ),
        "fine": torch.empty(
            (1, len(rgb_imgfiles), 24, *dims["fine"]), dtype=torch.float16
        ),
    }
    cam_positions = torch.from_numpy(cam_positions).cuda()[None]
    for i in range(len(rgb_imgfiles)):
        rgb_img = data.load_rgb_imgs([rgb_imgfiles[i]], imheight, imwidth)
        rgb_img = torch.from_numpy(rgb_img).cuda()[None]
        cur_proj_mats = {k: v[:, i, None] for k, v in proj_mats.items()}
        cur_feats_2d = model.sdfformer.get_img_feats(
            rgb_img, cur_proj_mats, cam_positions[:, i, None]
        )
        for resname in feats_2d:
            feats_2d[resname][0, i] = cur_feats_2d[resname][0, 0].cpu()
    return feats_2d


def inference(model, info_file, outfile, n_imgs, cropsize, voxel_size=0.04):
    rgb_imgfiles, depth_imgfiles, pose, intr = load_scene(info_file)
    test_img = imageio.imread(rgb_imgfiles[0])
    imheight, imwidth, _ = test_img.shape

    scene_minbound, scene_maxbound = get_scene_bounds(
        pose, intr, imheight, imwidth, frustum_depth=4
    )

    pose_w2c = np.linalg.inv(pose)
    tiles = get_tiles(
        scene_minbound,
        scene_maxbound,
        cropsize_voxels_fine=np.array(cropsize),
        voxel_size_fine=voxel_size,
    )

    # pre-select views for each tile
    tiles = frame_selection(
        tiles, pose, intr, imheight, imwidth, n_imgs=n_imgs, rmin_deg=15, tmin=0.1
    )

    # drop the frames that weren't selected for any tile, re-index the selected frame indicies
    selected_frame_inds = np.unique(
        np.concatenate([tile["frame_inds"] for tile in tiles])
    )
    all_frame_inds = np.arange(len(pose))
    frame_reindex = np.full(len(all_frame_inds), 100_000)
    frame_reindex[selected_frame_inds] = np.arange(len(selected_frame_inds))
    for tile in tiles:
        tile["frame_inds"] = frame_reindex[tile["frame_inds"]]
    pose_w2c = pose_w2c[selected_frame_inds]
    pose = pose[selected_frame_inds]
    rgb_imgfiles = np.array(rgb_imgfiles)[selected_frame_inds]

    factors = np.array([1 / 16, 1 / 8, 1 / 4])
    proj_mats = data.get_proj_mats(intr, pose_w2c, factors)
    proj_mats = {k: torch.from_numpy(v)[None].cuda() for k, v in proj_mats.items()}
    img_feats = get_img_feats(
        model,
        imheight,
        imwidth,
        proj_mats,
        rgb_imgfiles,
        cam_positions=pose[:, :3, 3],
    )
    for resname, res in model.sdfformer.resolutions.items():

        # populate feature volume independently for each tile
        for tile in tiles:
            voxel_coords = tile["voxel_coords"].cuda()
            voxel_batch_inds = torch.zeros(
                len(voxel_coords), dtype=torch.int64, device="cuda"
            )

            cur_img_feats = img_feats[resname][:, tile["frame_inds"]].cuda()
            cur_proj_mats = proj_mats[resname][:, tile["frame_inds"]]

            featheight, featwidth = img_feats[resname].shape[-2:]
            bp_uv, bp_depth, bp_mask = model.sdfformer.project_voxels(
                voxel_coords,
                voxel_batch_inds,
                cur_proj_mats.transpose(0, 1),
                featheight,
                featwidth,
            )
            bp_data = {
                "voxel_batch_inds": voxel_batch_inds,
                "bp_uv": bp_uv,
                "bp_depth": bp_depth,
                "bp_mask": bp_mask,
            }
            del voxel_coords
            torch.cuda.empty_cache()
            bp_feats, proj_occ_logits = model.sdfformer.back_project_features(
                bp_data,
                cur_img_feats.transpose(0, 1),
                model.sdfformer.mv_fusion[resname],
            )
            bp_feats = model.sdfformer.layer_norms[resname](bp_feats)

            tile["voxel_features"] = torch.cat(
                (tile["voxel_features"], bp_feats.cpu(), tile["voxel_logits"]),
                dim=-1,
            )
            torch.cuda.empty_cache()

        # combine all tiles into one sparse tensor & run convolution
        voxel_inds = torch.cat([tile["voxel_inds"] for tile in tiles], dim=0)
        voxel_batch_inds = torch.zeros((len(voxel_inds), 1), dtype=torch.int32)
        if resname == 'coarse':
            voxel_dim_16 = voxel_inds[-1][:3] + 1
        voxel_dim = (voxel_dim_16*(model.sdfformer.resolutions['coarse']/res)).int().tolist()

        voxel_features = spconv.SparseConvTensor(
            torch.cat([tile["voxel_features"] for tile in tiles], dim=0).cuda(), 
            torch.cat([voxel_batch_inds, voxel_inds], dim=-1).cuda(), 
            voxel_dim, batch_size=1)

        print('== voxel_features:', voxel_features.features.shape, voxel_dim)
        torch.cuda.empty_cache()
        voxel_features = model.sdfformer.net3d[resname](voxel_features, voxel_dim, res, hash_size=4*voxel_features.features.shape[0])
        voxel_logits = model.sdfformer.output_layers[resname](voxel_features)

        if resname in ["coarse", "medium"]:
            # sparsify & upsample
            occupancy = voxel_logits.features.squeeze(1) > 0
            if not torch.any(occupancy):
                raise Exception("um")
            voxel_features = model.sdfformer.upsampler.upsample_feats(
                voxel_features.features[occupancy]
            )
            voxel_inds = model.sdfformer.upsampler.upsample_inds(bxyz2xyzb(voxel_logits.indices[occupancy]))
            voxel_logits = model.sdfformer.upsampler.upsample_feats(
                voxel_logits.features[occupancy]
            )
            voxel_features = voxel_features.cpu()
            voxel_inds = voxel_inds.cpu()
            voxel_logits = voxel_logits.cpu()

            # split back up into tiles
            for tile in tiles:
                tile["origin_ind"] *= 2
                tile["maxbound_ind"] *= 2

                tile_voxel_mask = (
                    (voxel_inds[:, 0] >= tile["origin_ind"][0])
                    & (voxel_inds[:, 1] >= tile["origin_ind"][1])
                    & (voxel_inds[:, 2] >= tile["origin_ind"][2])
                    & (voxel_inds[:, 0] < tile["maxbound_ind"][0])
                    & (voxel_inds[:, 1] < tile["maxbound_ind"][1])
                    & (voxel_inds[:, 2] < tile["maxbound_ind"][2])
                )

                tile["voxel_inds"] = voxel_inds[tile_voxel_mask, :3]
                tile["voxel_features"] = voxel_features[tile_voxel_mask]
                tile["voxel_logits"] = voxel_logits[tile_voxel_mask]
                tile["voxel_coords"] = tile["voxel_inds"] * (
                    res / 2
                ) + scene_minbound.astype(np.float32)

    tsdf_vol = utils.to_vol(
        voxel_logits.indices[:, 1:].cpu().numpy(),
        1.05 * torch.tanh(voxel_logits.features).squeeze(-1).cpu().numpy(),
    )
    mesh = utils.to_mesh(
        -tsdf_vol,
        voxel_size=voxel_size,
        origin=scene_minbound,
        level=0,
        mask=~np.isnan(tsdf_vol),
    )
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--use-proj-occ", default=True, type=bool)
    parser.add_argument("--n-imgs", default=60, type=int)
    parser.add_argument("--cropsize", default=96, type=int)
    parser.add_argument("--voxel_size", default=0.04, type=float)
    args = parser.parse_args()

    pl.seed_everything(0)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cropsize = (args.cropsize, args.cropsize, 48)

    with torch.cuda.amp.autocast():

        info_files = utils.load_info_files(config["scannet_dir"], args.split)
        model = load_model(args.ckpt, args.use_proj_occ, config)
        for info_file in tqdm.tqdm(info_files):

            scene_name = os.path.basename(os.path.dirname(info_file))
            outdir = os.path.join(args.outputdir, scene_name)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, "prediction.ply")

            if os.path.exists(outfile):
                print(outfile, 'exists, skipping')
                continue

            print('== Infer', scene_name)
            try:
                with torch.no_grad():
                    mesh = inference(model, info_file, outfile, args.n_imgs, cropsize, args.voxel_size)
                o3d.io.write_triangle_mesh(outfile, mesh)
            except Exception as e:
                print(e)
