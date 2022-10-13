import os
import sys
import torch
import lietorch
import cv2
import numpy as np
import argparse
import open3d as o3d
import droid_backends


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction_path",
                        help="path to saved reconstruction")
    parser.add_argument("--max_abs_error", default=0.01,
                        help="Depth consistency threshold in metter")
    parser.add_argument("--min_neighbor_count", default=3,
                        help="Number of neighbor keyframes which predict consistent depth")
    args = parser.parse_args()

    folder = f"reconstructions/{args.reconstruction_path}"
    tstamps = torch.as_tensor(np.load(f"{folder}/tstamps.npy")).cuda()
    images = torch.as_tensor(np.load(f"{folder}/images.npy"))
    disps = torch.as_tensor(np.load(f"{folder}/disps.npy")).cuda()
    poses = torch.as_tensor(np.load(f"{folder}/poses.npy")).cuda()
    intrinsics = torch.as_tensor(np.load(f"{folder}/intrinsics.npy")).cuda()

    # masks
    index = torch.arange(tstamps.shape[0], device='cuda')
    thresh = args.min_neighbor_count * torch.ones_like(disps.mean(dim=[1, 2]))
    counts = droid_backends.depth_filter(
        poses, disps, intrinsics[0], index, thresh)
    masks = ((counts >= args.max_abs_error) & (
        disps > .5*disps.mean(dim=[1, 2], keepdim=True)))
    masks = masks.reshape(-1)

    # points
    points = droid_backends.iproj(lietorch.SE3(
        poses).inv().data, disps, intrinsics[0]).cpu()
    points_mask = points.reshape(-1, 3)[masks].numpy()

    # colors
    images = images[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
    colors_mask = images.reshape(-1, 3)[masks].numpy()

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_mask)
    pc.colors = o3d.utility.Vector3dVector(colors_mask)

    outfile = f"{folder}/pc_count{args.min_neighbor_count}_abs{args.max_abs_error*100}cm.ply"
    o3d.io.write_point_cloud(outfile, pc)
    print(f"saved to {outfile}")
