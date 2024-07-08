import os
import cv2
import json
import torch
import numpy as np
import kornia as K
import kornia.feature as KF
from plyfile import PlyData, PlyElement
from typing import Union
from shutil import rmtree
import argparse

def read_image(path: str, bg_color = (255, 255, 255)) -> np.ndarray:
    image = cv2.imread(path, -1)
    if image.ndim == 3 and image.shape[-1] == 4:
        mask = image[..., -1].astype(np.bool_)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image[~mask] = bg_color
        image = image.astype(np.float32) / 255.
        return image, mask

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.

    return image

def write_points(path: str, points: Union[np.ndarray, torch.FloatTensor]):
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if points.shape[1] == 6:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        red, green, blue = points[:, 3], points[:, 4], points[:, 5]
        vertex = np.empty(len(points), dtype=dtype)
        vertex['x'] = x.astype('f4')
        vertex['y'] = y.astype('f4')
        vertex['z'] = z.astype('f4')
        vertex['red'] = np.clip(red*255, 0, 255).astype('u1')
        vertex['green'] = np.clip(green*255, 0, 255).astype('u1')
        vertex['blue'] = np.clip(blue*255, 0, 255).astype('u1')
    else:
        vertex = np.empty(len(points), dtype=dtype)
        vertex['x'] = x.astype('f4')
        vertex['y'] = y.astype('f4')
        vertex['z'] = z.astype('f4')
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(path)

def _check_frustum_overlap(poses: torch.FloatTensor, intrinsic: torch.FloatTensor,
                           H: int, W: int, near: float, far: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frustum_points = np.array([[.5, .5, 1.],
                               [W-1+.5, .5, 1.],
                               [.5, H-1+.5, 1.],
                               [W-1+.5, W-1+.5, 1.]])
    frustum_points = torch.FloatTensor(frustum_points).to(device)
    frustum_points = torch.cat([frustum_points*near, frustum_points*far], dim=0)
    intrinsic = intrinsic.to(device)
    pairs = []
    for i in range(len(poses)-1):
        src_c2w = poses[i].to(device)      # (4, 4)
        dst_c2w = poses[i+1:].to(device)   # (N, 4, 4)
        
        # project frustum points to world coordinate
        prj = (torch.linalg.inv(intrinsic) @ frustum_points.T).T
        src_R, src_t = src_c2w[0:3, 0:3], src_c2w[0:3, 3:]
        prj = (src_R @ prj.T + src_t).T
        # project frustum points to other camera coordinates
        dst_w2c = torch.linalg.inv(dst_c2w)
        dst_Rs, dst_ts = dst_w2c[:, 0:3, 0:3], dst_w2c[:, 0:3, 3:]
        prj = (dst_Rs @ prj.T + dst_ts).transpose(1, 2) # (N, 8, 3)
        # project frustum points to other pixel coordinates
        prj = torch.einsum('DC, NLC -> NLD', intrinsic, prj)
        
        # check frustum points in pixel plane
        u, v, d = prj[..., 0], prj[..., 1], prj[..., 2]
        u /= torch.clamp(torch.abs(d), 1e-6)
        v /= torch.clamp(torch.abs(d), 1e-6)
        mask = (d > 0.) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        mask = mask.sum(dim=1) > 0
        idxs = torch.arange(i+1, len(poses), device=mask.device)[mask]

        for j in idxs:
            pairs.append((i, j.item()))

    pairs = torch.LongTensor(np.array(pairs))

    return pairs

def _loftr_keypoint_matcher(images: torch.FloatTensor, pairs: torch.LongTensor,
                            min_match_num: int = 20, batch_size: int = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = KF.LoFTR(pretrained='indoor').to(device)
    batch_indexes = []
    keypoints0 = []
    keypoints1 = []
    
    for i in range(0, len(pairs), batch_size):
        img0 = images[pairs[i:i+batch_size, 0]].to(device).permute(0, 3, 1, 2)
        img1 = images[pairs[i:i+batch_size, 1]].to(device).permute(0, 3, 1, 2)
        input_dict = {
            'image0': K.color.rgb_to_grayscale(img0),
            'image1': K.color.rgb_to_grayscale(img1),
        }
        with torch.inference_mode():
            correspondences = matcher(input_dict)
            batch_indexes.append(correspondences['batch_indexes'] + i)
            keypoints0.append(correspondences['keypoints0'])
            keypoints1.append(correspondences['keypoints1'])
    
    batch_indexes = torch.cat(batch_indexes, dim=0)
    keypoints0 = torch.cat(keypoints0, dim=0)
    keypoints1 = torch.cat(keypoints1, dim=0)
    
    pair_idxs, counts = torch.unique(batch_indexes, return_counts=True)
    pair_idxs = pair_idxs[counts >= min_match_num]
    
    matches = []
    for i in pair_idxs:
        mask = batch_indexes == i
        kpts0 = keypoints0[mask].cpu().numpy()
        kpts1 = keypoints1[mask].cpu().numpy()
        img_pair = pairs[i].cpu().numpy()
        matches.append({
            'img0_idx': img_pair[0],
            'img1_idx': img_pair[1],
            'kpts0': kpts0,
            'kpts1': kpts1
        })
    return matches

def _get_rays(pts2d, poses, intrisic):
    z = np.ones((pts2d.shape[0], 1), dtype=np.float32)
    pts = np.concatenate([pts2d, z], axis=-1)
    
    pts = np.linalg.inv(intrisic) @ pts.T
    pts = poses[:3, :3] @ pts
    pts = pts.T + poses[:3, 3]  # (N, 3)

    rays_o = np.broadcast_to(poses[:3, 3], pts.shape)
    rays_d = pts - rays_o

    return rays_o, rays_d

def _distance_point2line(pts3d, rays_o, rays_d):
    # pts3d: (N, 3)
    # rays_o: (N, 3)
    # rays_d: (N, 3)
    # return: (N, )
    # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    dist = np.linalg.norm(np.cross(rays_d, pts3d - rays_o), axis=-1) / np.linalg.norm(rays_d, axis=-1)
    depth = np.sum((pts3d - rays_o) * rays_d, axis=-1, keepdims=True) / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    cross_pts = rays_o + rays_d * depth[:, None]
    # # valid
    # v = np.linalg.norm(pts3d - cross_pts, axis=-1)
    # err = v - dist
    return dist, cross_pts, depth.reshape(-1)

def _line_min_distance(o0, d0, o1, d1):
    A = np.stack([d0.reshape(-1, 3), d1.reshape(-1, 3)], axis=-1)
    b = o1.reshape(-1, 3, 1) - o0.reshape(-1, 3, 1)

    # batch least square
    A = torch.FloatTensor(A).cuda()
    b = torch.FloatTensor(b).cuda()
    
    X = torch.linalg.lstsq(A, b).solution
    t0 = X[:, 0, 0].cpu().numpy()
    t1 = -X[:, 1, 0].cpu().numpy()

    p0 = o0 + d0 * t0[:, None]
    p1 = o1 + d1 * t1[:, None]
    min_dist = np.linalg.norm(p0 - p1, axis=-1)
    min_pts = (p0 + p1) / 2
    return min_dist, t0, t1, p0, p1, min_pts

def _filter_by_geometric_constraint(matches: dict, poses: np.ndarray, intrinsic: np.ndarray, near, far,
                                    pl_threshold: float = 0.05, min_match_num: int = 20):
    filtered_matches = []
    for m in matches:
        img0_idx = m['img0_idx']
        img1_idx = m['img1_idx']
        kpts0 = m['kpts0'].astype(np.float64)
        kpts1 = m['kpts1'].astype(np.float64)

        # prj_mat0 = (intrinsic @ np.linalg.inv(poses[img0_idx])[:3]).astype(np.float64)
        # prj_mat1 = (intrinsic @ np.linalg.inv(poses[img1_idx])[:3]).astype(np.float64)
        # pts3d = cv2.triangulatePoints(prj_mat0, prj_mat1, kpts0.T, kpts1.T)
        # pts3d = (pts3d / pts3d[3, :]).astype(np.float32) # (4, N)

        rays_o0, rays_d0 = _get_rays(kpts0, poses[img0_idx], intrinsic)
        rays_o1, rays_d1 = _get_rays(kpts1, poses[img1_idx], intrinsic)

        # 这里要用相机位置, 不能用rays_o
        real_o0 = np.broadcast_to(poses[img0_idx][:3, 3], rays_o0.shape)
        real_o1 = np.broadcast_to(poses[img1_idx][:3, 3], rays_o1.shape)
        min_dist, depth0, depth1, pts3d0, pts3d1, min_pts3d = _line_min_distance(real_o0, rays_d0, real_o1, rays_d1)
        mask = (min_dist <= pl_threshold) & (depth0 >= near) & (depth0 <= far) & (depth1 >= near) & (depth1 <= far)

        if mask.sum() > 0 and mask.sum() >= min_match_num:
            filtered_matches.append({
                'img0_idx': img0_idx,
                'img1_idx': img1_idx,
                'kpts0': kpts0[mask].astype(np.float32),
                'kpts1': kpts1[mask].astype(np.float32),
                'depth0': depth0[mask].astype(np.float32),
                'depth1': depth1[mask].astype(np.float32),
                'pts3d0': pts3d0[mask].astype(np.float32),
                'pts3d1': pts3d1[mask].astype(np.float32),
                'min_pts3d': min_pts3d[mask].astype(np.float32),
            })

    return filtered_matches

def _filter_by_epipoplar_constraint(matches: dict, poses: np.ndarray, intrinsic: np.ndarray,
                                    threshold: float = 3., min_match_num: int = 20):
    filtered_matches = []
    for m in matches:
        img0_idx = m['img0_idx']
        img1_idx = m['img1_idx']
        kpts0 = m['kpts0']
        kpts1 = m['kpts1']
        kpts0_h = np.concatenate([kpts0, np.ones_like(kpts0[:, :1])], axis=-1)
        kpts1_h = np.concatenate([kpts1, np.ones_like(kpts1[:, :1])], axis=-1)
        
        P_c0_w = poses[img0_idx]
        P_c1_w = poses[img1_idx]
        P_c0_c1 = np.linalg.inv(P_c1_w) @ P_c0_w
        R, t = P_c0_c1[0:3, 0:3], P_c0_c1[0:3, 3]
        
        t_sy = np.array([[0., -t[2], t[1]],
                         [t[2], 0., -t[0]],
                         [-t[1], t[0], 0]], dtype=np.float32)
        E = t_sy @ R
        F = np.linalg.inv(intrinsic).T @ E @ np.linalg.inv(intrinsic)
        # p1.T @ F @ p0 = 0
        epipolar_line = (F @ kpts0_h.T).T
        epipolar_line_ = epipolar_line / np.linalg.norm(epipolar_line[:, :2], axis=1, keepdims=True)
        err = np.abs(np.sum(kpts1_h * epipolar_line_, axis=1))
        select_idxs, = np.where(err <= threshold)
        
        if len(select_idxs) >= min_match_num:
            filtered_matches.append({
                'img0_idx': img0_idx,
                'img1_idx': img1_idx,
                'kpts0': kpts0[select_idxs],
                'kpts1': kpts1[select_idxs]
            })
    return filtered_matches

def loftr_pair_estimator(image_list: list, pose_list: list, 
                         fx: float, fy: float,
                         cx: float, cy: float,
                         near: float, far: float,
                         min_match_num: int = 20,
                         pl_threshold: float = 0.1,
                         batch_size: int = 4):
    if len(image_list) < batch_size:
        batch_size = len(image_list)

    images = torch.FloatTensor(np.stack(image_list, axis=0))
    poses = torch.FloatTensor(np.stack(pose_list, axis=0))
    intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)
    intrinsic = torch.FloatTensor(intrinsic)
    height, width = images.shape[1:3]

    print("computing frustum overlap ...")
    pairs = _check_frustum_overlap(poses, intrinsic, height, width, near, far)
    print("image matching...")
    matches = _loftr_keypoint_matcher(images, pairs, min_match_num, batch_size)
    print("filtering outliers ...")
    matches = _filter_by_epipoplar_constraint(matches, poses.cpu().numpy(), intrinsic.cpu().numpy(),
                                              3.0, min_match_num) # 重投影误差固定3.0
    matches = _filter_by_geometric_constraint(matches, poses.cpu().numpy(), intrinsic.cpu().numpy(), near, far,
                                              pl_threshold, min_match_num)
    return matches


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/P2NeRF/DDP')
    parser.add_argument('--out_dir', type=str, default='data/P2NeRF/prior/DDP')
    parser.add_argument('--min_match_num', type=int, default=20)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()
    return args


BLENDER2COLMAP = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)


args = get_args()

for scan_name in os.listdir(args.data_dir):
    scan_dir = os.path.join(args.data_dir, scan_name)
    for split in ('train',):
        print(f'Process {scan_name} {split} ...')
        kpts_dir = os.path.join(args.out_dir, f'{scan_name}/{split}/keypoints_pair')
        vis_kpts_dir = os.path.join(args.out_dir, f'{scan_name}/{split}/vis_keypoints_pair')
        if os.path.exists(kpts_dir):
            print(f'Remove {kpts_dir} ...')
            rmtree(kpts_dir)
        if os.path.exists(vis_kpts_dir):
            print(f'Remove {vis_kpts_dir} ...')
            rmtree(vis_kpts_dir)
        os.makedirs(kpts_dir, exist_ok=True)
        os.makedirs(vis_kpts_dir, exist_ok=True)
        
        image_name_list = []
        image_list = []
        pose_list = []
        
        with open(os.path.join(scan_dir, f'transforms_{split}.json'), 'r') as f:
            meta = json.load(f)
            near, far = meta['near'], meta['far']
            
            for frame in meta['frames']:
                fx, fy = frame['fx'], frame['fy']
                cx, cy = frame['cx'], frame['cy']
                image = read_image(os.path.join(scan_dir, frame['file_path']))
                pose = np.array(frame['transform_matrix'], dtype=np.float32) @ BLENDER2COLMAP
                image_name = os.path.splitext(os.path.basename(frame['file_path']))[0]
                
                if args.factor > 1:
                    fx = fx / args.factor
                    fy = fy / args.factor
                    cx = cx / args.factor
                    cy = cy / args.factor
                    raw_H, raw_W = image.shape[:2]
                    image = cv2.resize(image, (raw_W // args.factor, raw_H // args.factor))
                else:
                    raw_H, raw_W = image.shape[:2]

                image_name_list.append(image_name)
                image_list.append(image)
                pose_list.append(pose)
                
        matches = loftr_pair_estimator(image_list, pose_list, fx, fy, cx, cy, near, far,
                                       args.min_match_num, args.threshold, args.batch_size)
        
        all_points3d = []
        for m in matches:
            name0 = image_name_list[m['img0_idx']]
            name1 = image_name_list[m['img1_idx']]
            kpts0 = m['kpts0'] # xy format
            kpts1 = m['kpts1']
            depth0 = m['depth0']
            depth1 = m['depth1']
            pts3d0 = m['pts3d0']
            pts3d1 = m['pts3d1']
            min_pts3d = m['min_pts3d']
            print(f'{name0}-{name1} keypoints: {kpts0.shape[0]}')

            # 从image0取颜色
            pts_color0 = image_list[m['img0_idx']][kpts0[:, 1].astype(np.int32), kpts0[:, 0].astype(np.int32)]
            all_points3d.append(np.concatenate([pts3d0, pts_color0], axis=-1))
            # 从image1取颜色
            pts_color1 = image_list[m['img1_idx']][kpts1[:, 1].astype(np.int32), kpts1[:, 0].astype(np.int32)]
            all_points3d.append(np.concatenate([pts3d1, pts_color1], axis=-1))
            min_pts3d = np.concatenate([min_pts3d, pts_color0], axis=-1)
            write_points(os.path.join(vis_kpts_dir, f'{name0}-{name1}_min_pts.txt'), min_pts3d)

            if args.factor > 1:
                kpts0 = (kpts0 * args.factor).astype(np.int32)
                kpts1 = (kpts1 * args.factor).astype(np.int32)
            else:
                kpts0 = kpts0.astype(np.int32)
                kpts1 = kpts1.astype(np.int32)

            path = os.path.join(kpts_dir, f'{name0}-{name1}.npz')
            np.savez(path, name0=name0, name1=name1,
                     kpts0=kpts0, kpts1=kpts1,
                     depth0=depth0, depth1=depth1)
            
            img0 = cv2.resize(np.copy(image_list[m['img0_idx']]), (raw_W, raw_H))
            img1 = cv2.resize(np.copy(image_list[m['img1_idx']]), (raw_W, raw_H))
            match_image = np.concatenate([img0, img1], axis=1)
            match_image = (match_image * 255).astype(np.uint8)
            match_image = cv2.cvtColor(match_image, cv2.COLOR_RGB2BGR)
            w = img0.shape[1]
            for i in range(kpts0.shape[0]):
                color = np.random.randint(0, 255, size=3).tolist()
                cv2.circle(match_image, (kpts0[i, 0], kpts0[i, 1]), 2, color, -1)
                cv2.circle(match_image, (kpts1[i, 0] + w, kpts1[i, 1]), 2, color, -1)
                cv2.line(match_image, (kpts0[i, 0], kpts0[i, 1]), (kpts1[i, 0] + w, kpts1[i, 1]), color, 1)
            cv2.imwrite(os.path.join(vis_kpts_dir, f'{name0}-{name1}.png'), match_image)

        print(f'Process {scan_name} {split} done!')
        all_points3d = np.concatenate(all_points3d, axis=0)
        write_points(os.path.join(args.out_dir, f'{scan_name}/{split}/keypoints3d.ply'), all_points3d)
