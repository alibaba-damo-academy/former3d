import open3d as o3d
import numpy as np


def eval_mesh(pred_mesh, gt_mesh, threshold=0.05, down_sample=0.02):
    """Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points
    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(pred_mesh.vertices))
    )
    pcd_trgt = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(gt_mesh.vertices))
    )

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist_trgt_to_pred = nn_correspondance(verts_pred, verts_trgt)
    _, dist_pred_to_trgt = nn_correspondance(verts_trgt, verts_pred)
    dist_trgt_to_pred = np.array(dist_trgt_to_pred)
    dist_pred_to_trgt = np.array(dist_pred_to_trgt)

    precision = np.mean((dist_pred_to_trgt < threshold).astype("float"))
    recall = np.mean((dist_trgt_to_pred < threshold).astype("float"))
    fscore = 2 * precision * recall / (precision + recall)
    metrics = {
        "comp": np.mean(dist_trgt_to_pred),
        "acc": np.mean(dist_pred_to_trgt),
        "prec": precision,
        "recall": recall,
        "fscore": fscore,
    }
    return metrics


def nn_correspondance(verts1, verts2):
    """for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def project_to_mesh(from_mesh, to_mesh, attribute, dist_thresh=None):
    """Transfers attributs from from_mesh to to_mesh using nearest neighbors
    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.
    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)
    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attribute] = np.zeros((0), dtype=np.uint8)
        to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = from_mesh.vertex_attributes[attribute]
    pred_colors = from_mesh.visual.vertex_colors

    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
    matched_colors = np.zeros((to_mesh.vertices.shape[0], 4), dtype=np.uint8)

    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0] < dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i] = pred_colors[inds[0]]

    mesh = to_mesh.copy()
    mesh.vertex_attributes[attribute] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh


def eval_depth(depth_pred, depth_trgt):
    """Computes 2d metrics between two depth maps

    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth
    Returns:
        Dict of metrics
    """
    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype("float")
    r2 = (thresh < 1.25 ** 2).astype("float")
    r3 = (thresh < 1.25 ** 3).astype("float")

    metrics = {}
    metrics["AbsRel"] = np.mean(abs_rel)
    metrics["AbsDiff"] = np.mean(abs_diff)
    metrics["SqRel"] = np.mean(sq_rel)
    metrics["RMSE"] = np.sqrt(np.mean(sq_diff))
    metrics["LogRMSE"] = np.sqrt(np.mean(sq_log_diff))
    metrics["r1"] = np.mean(r1)
    metrics["r2"] = np.mean(r2)
    metrics["r3"] = np.mean(r3)
    metrics["complete"] = np.mean(mask1.astype("float"))

    return metrics
