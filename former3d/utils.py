import functools
import glob
import itertools
import os

import numba
import numpy as np
import open3d as o3d
import skimage.measure
import wandb


def log_transform(x, shift=1):
    # https://github.com/magicleap/Atlas
    """rescales TSDF values to weight voxels near the surface more than close
    to the truncation distance"""
    return x.sign() * (1 + x.abs() / shift).log()


def to_vol(inds, vals):
    dims = np.max(inds, axis=0) + 1
    vol = np.ones(dims) * np.nan
    vol[inds[:, 0], inds[:, 1], inds[:, 2]] = vals
    return vol


def to_mesh(vol, voxel_size=1, origin=np.zeros(3), level=0, mask=None):
    verts, faces, _, _ = skimage.measure.marching_cubes(vol, level=level, mask=mask)
    verts *= voxel_size
    verts += origin

    bad_face_inds = np.any(np.isnan(verts[faces]), axis=(1, 2))
    faces = faces[~bad_face_inds]

    bad_vert_inds = np.any(np.isnan(verts), axis=-1)
    reindex = np.cumsum(~bad_vert_inds) - 1
    faces = reindex[faces]
    verts = verts[~bad_vert_inds]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )
    mesh.compute_vertex_normals()
    return mesh


@numba.jit(nopython=True)
def remove_redundant(poses, rmin_deg, tmin):
    cos_t_max = np.cos(rmin_deg * np.pi / 180)
    frame_inds = np.arange(len(poses))
    selected_frame_inds = [frame_inds[0]]
    for frame_ind in frame_inds[1:]:
        prev_pose = poses[selected_frame_inds[-1]]
        candidate_pose = poses[frame_ind]
        cos_t = np.sum(prev_pose[:3, 2] * candidate_pose[:3, 2])
        tdist = np.linalg.norm(prev_pose[:3, 3] - candidate_pose[:3, 3])
        if tdist > tmin or cos_t < cos_t_max:
            selected_frame_inds.append(frame_ind)
    return selected_frame_inds


def frame_selection(
    poses,
    intr,
    imwidth,
    imheight,
    sample_pts,
    tmin,
    rmin_deg,
    n_imgs,
):
    # select randomly among views that see at least one sample point

    intr4x4 = np.eye(4, dtype=np.float32)
    intr4x4[:3, :3] = intr

    xyz = np.concatenate(
        (sample_pts, np.ones((len(sample_pts), 1), dtype=sample_pts.dtype)), axis=-1
    )
    uv = intr4x4 @ np.linalg.inv(poses) @ xyz.T
    z = uv[:, 2]
    z_valid = z > 1e-10
    z[~z_valid] = 1
    uv = uv[:, :2] / z[:, None]
    valid = (
        (uv[:, 0] > 0)
        & (uv[:, 0] < imwidth)
        & (uv[:, 1] > 0)
        & (uv[:, 1] < imheight)
        & z_valid
    )
    intersections = np.sum(valid, axis=-1)
    intersect_inds = np.argwhere(intersections > 0).flatten()

    frame_inds = np.arange(len(poses), dtype=np.int32)

    if n_imgs is None:
        score = intersections[intersect_inds]
        selected_frame_inds = frame_inds[intersect_inds]
    elif len(intersect_inds) >= n_imgs:
        selected_frame_inds = np.random.choice(
            intersect_inds, size=n_imgs, replace=False
        )
        score = intersections[selected_frame_inds]
    else:
        not_intersect_inds = np.argwhere(intersections == 0).flatten()
        n_needed = n_imgs - len(intersect_inds)
        extra_inds = np.random.choice(not_intersect_inds, size=n_needed, replace=False)
        selected_frame_inds = np.concatenate((intersect_inds, extra_inds))
        score = np.concatenate((intersections[intersect_inds], np.zeros(n_needed)))

    return (selected_frame_inds, score)


def load_info_files(scannet_dir, split):
    with open(os.path.join(scannet_dir, f"scannetv2_{split}.txt"), "r") as f:
        scene_names = f.read().split()
    scan_dir = "scans" if split in ["train", "val"] else "scans_test"
    info_files = sorted(glob.glob(os.path.join(scannet_dir, scan_dir, "*/info.json")))
    info_files = [
        f for f in info_files if os.path.basename(os.path.dirname(f)) in scene_names
    ]
    return info_files


def visualize(o3d_geoms):
    visibility = [True] * len(o3d_geoms)

    def toggle_geom(vis, geom_ind):
        if visibility[geom_ind]:
            vis.remove_geometry(o3d_geoms[geom_ind], reset_bounding_box=False)
            visibility[geom_ind] = False
        else:
            vis.add_geometry(o3d_geoms[geom_ind], reset_bounding_box=False)
            visibility[geom_ind] = True

    callbacks = {}
    for i in range(len(o3d_geoms)):
        callbacks[ord(str(i + 1))] = functools.partial(toggle_geom, geom_ind=i)
    o3d.visualization.draw_geometries_with_key_callbacks(o3d_geoms, callbacks)
