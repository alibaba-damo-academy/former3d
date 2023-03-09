import argparse
import glob
import json
import os
import yaml

os.environ["PYOPENGL_PLATFORM"] = "egl"

import imageio
import open3d as o3d
import numpy as np
import pyrender
import trimesh
import tqdm
import multiprocessing
import pycuda.driver as cuda

from former3d import data, eval_utils, utils
from former3d import tsdf_fusion


Multi_Mode = 'Thread'  # 'Thread', 'Process'
if Multi_Mode == 'Thread':
    cuda.init()


class Renderer:
    """OpenGL mesh renderer

    Used to render depthmaps from a mesh for 2d evaluation
    """

    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
        )
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()


def worker(arguments):
    config, scan_dir, pred_plyfile = arguments
    savefile = os.path.join(os.path.dirname(pred_plyfile), "metrics.json")

    # ##### skip if computed before
    # if os.path.exists(savefile):
    #     return
    
    if Multi_Mode == 'Thread':
        ctx = cuda.Device(0).make_context()
        
    scene_name = os.path.basename(os.path.dirname(pred_plyfile))
    pred_mesh_o3d = o3d.io.read_triangle_mesh(pred_plyfile)
    gt_mesh_o3d = o3d.io.read_triangle_mesh(
        os.path.join(
            scan_dir,
            scene_name,
            scene_name + "_vh_clean_2.ply",
        )
    )
    gt_mesh_o3d.compute_vertex_normals()
    gt_tsdf_vol, origin, voxel_size = data.load_tsdf(config["tsdf_dir"], scene_name)
    tsdf_volume = tsdf_fusion.TSDFVolume(
        np.c_[origin, origin + np.array(gt_tsdf_vol.shape) * voxel_size],
        voxel_size,
        use_gpu=True,
        margin=3,
    )
    pred_mesh = trimesh.Trimesh(
        vertices=np.asarray(pred_mesh_o3d.vertices),
        faces=np.asarray(pred_mesh_o3d.triangles),
    )
    gt_mesh = trimesh.Trimesh(
        vertices=np.asarray(gt_mesh_o3d.vertices),
        faces=np.asarray(gt_mesh_o3d.triangles),
    )
    renderer = Renderer()
    pred_mesh_opengl = renderer.mesh_opengl(pred_mesh)
    gt_mesh_opengl = renderer.mesh_opengl(gt_mesh)

    info_file = os.path.join(scan_dir, scene_name, "info.json")
    with open(info_file, "r") as f:
        info = json.load(f)
    scene_name = info["scene"]
    rgb_imgfiles = [frame["filename_color"] for frame in info["frames"]]
    depth_imgfiles = [frame["filename_depth"] for frame in info["frames"]]
    pose = np.stack([frame["pose"] for frame in info["frames"]], axis=0).astype(
        np.float32
    )
    intr = np.array(info["intrinsics"])

    depth_metrics = {}

    for i in tqdm.trange(len(pose)):
        depth_gt = imageio.imread(depth_imgfiles[i]).astype(np.float32) / 1000
        imheight, imwidth = depth_gt.shape

        _, depth_pred = renderer(imheight, imwidth, intr, pose[i], pred_mesh_opengl)

        for k, v in eval_utils.eval_depth(depth_pred, depth_gt).items():
            if k not in depth_metrics:
                depth_metrics[k] = []
            else:
                depth_metrics[k].append(v)

        _, gt_mask = renderer(imheight, imwidth, intr, pose[i], gt_mesh_opengl)
        depth_pred[gt_mask == 0] = 0

        rgb_img = imageio.imread(rgb_imgfiles[i])
        tsdf_volume.integrate(rgb_img, depth_pred, intr, pose[i])

    tsdf_vol, color_vol, weight_vol = tsdf_volume.get_volume()
    tsdf_vol = -tsdf_vol
    tsdf_vol[weight_vol == 0] = np.nan
    pred_mesh = utils.to_mesh(
        -tsdf_vol, voxel_size=0.04, level=0, origin=origin
    )

    # utils.visualize([pred_mesh, gt_mesh])

    depth_metrics = {k: np.nanmean(v) for k, v in depth_metrics.items()}
    mesh_metrics = eval_utils.eval_mesh(pred_mesh, gt_mesh)
    metrics = {**depth_metrics, **mesh_metrics}
    metrics = {k: float(v) for k, v in metrics.items()}
    
    # write metrics and trimmed mesh
    o3d.io.write_triangle_mesh(
        os.path.join(os.path.dirname(pred_plyfile), "trimmed_mesh.ply"), pred_mesh
    )
    with open(savefile, "w") as f:
        json.dump(metrics, f)

    print('== Finish ', savefile)
    if Multi_Mode == 'Thread':
        ctx.pop()
    try:
        renderer.delete()
    except Exception as e:
        print('!!! Exception:', e)

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_proc", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.split == "test":
        splitfile = os.path.join(config["scannet_dir"], "scannetv2_test.txt")
        scan_dir = os.path.join(config["scannet_dir"], "scans_test")
    elif args.split == "val":
        splitfile = os.path.join(config["scannet_dir"], "scannetv2_val.txt")
        scan_dir = os.path.join(config["scannet_dir"], "scans")

    with open(splitfile, "r") as f:
        scene_names = f.read().split()

    scene_dirs = [
        f
        for f in sorted(glob.glob(os.path.join(scan_dir, "*")))
        if os.path.basename(f) in scene_names
    ]

    pred_plyfiles = sorted(
        glob.glob(os.path.join(args.results_dir, "*", "prediction.ply"))
    )

    if args.n_proc == 0:
        results = []
        for pred_plyfile in tqdm.tqdm(pred_plyfiles):
            res = worker((config, scan_dir, pred_plyfile))
            results.append(res)
    else:
        if Multi_Mode == 'Process':
            pool = multiprocessing.Pool(processes = args.n_proc)
        elif Multi_Mode == 'Thread':
            pool = multiprocessing.pool.ThreadPool(args.n_proc)

        args_list = []
        for pred_plyfile in tqdm.tqdm(pred_plyfiles):
            args_list.append((config, scan_dir, pred_plyfile))
        results = pool.map(worker, args_list)
        pool.close()
        pool.join()

    report = {}
    for res in results:
        for k, v in res.items():
            if k not in report:
                report[k] = [v]
            else:
                report[k].append(v)
    print(f'== {len(results)} scene average ==')
    for k, v in report.items():
        v_mean = np.array(v).mean()
        print(k, "%.3f"%v_mean)