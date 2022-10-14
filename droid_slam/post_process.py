import torch
import lietorch
import numpy as np
import argparse
import open3d as o3d
import droid_backends
from pathlib import Path
from droid import Droid

def save_reconstruction(
        droid: Droid,
        traj_est: np.array,
        reconstruction_path: str
    ):
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps[:t].cpu().numpy()
    disps_up = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/disps_up.npy".format(reconstruction_path), disps_up)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    np.save("reconstructions/{}/trajectory.npy".format(reconstruction_path), traj_est)


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
    poses_cw = torch.as_tensor(np.load(f"{folder}/poses.npy")).cuda()
    intrinsics = torch.as_tensor(np.load(f"{folder}/intrinsics.npy")).cuda()
    traj = np.load(f"{folder}/trajectory.npy")

    # masks
    index = torch.arange(tstamps.shape[0], device='cuda')
    thresh = args.max_abs_error * torch.ones_like(disps.mean(dim=[1, 2]))
    counts = droid_backends.depth_filter(poses_cw, disps, intrinsics[0], index, thresh)
    masks = ((counts >= args.min_neighbor_count) & (
        disps > .5*disps.mean(dim=[1, 2], keepdim=True)))
    masks = masks.reshape(-1)

    # points
    poses_wc = lietorch.SE3(poses_cw).inv().data
    points = droid_backends.iproj(poses_wc, disps, intrinsics[0]).cpu()
    points_mask = points.reshape(-1, 3)[masks].numpy()

    # colors
    images = images[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
    colors_mask = images.reshape(-1, 3)[masks].numpy()

    # export pc
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_mask)
    pc.colors = o3d.utility.Vector3dVector(colors_mask)

    outfile = f"{folder}/pc_count{args.min_neighbor_count}_abs{args.max_abs_error*100}cm.ply"
    o3d.io.write_point_cloud(outfile, pc)
    print(f"saved to {outfile}")

    # export trajectory
    np.savetxt(f"{folder}/traj_kf.txt", poses_wc.cpu().numpy())       #  for evo evaluation
    np.savetxt(f"{folder}/traj_full.txt", traj)         #  for evo evaluation
    np.savetxt(f"{folder}/traj_full.poly", traj[:,:3])  #  for visualization in cloudcompare
    print("done")
