import os
import glob
import numpy as np
import mediapy as media
from PIL import Image
from matplotlib import cm
from tqdm import tqdm

def load_image(filename):
    with open(filename, 'rb') as f:
        image = np.array(Image.open(f), dtype=np.float32)
    return image

def create_video(image_dir, out_dir, out_name):
    num = len(glob.glob(os.path.join(image_dir, 'color_*.png')))
    depth_frame = load_image(os.path.join(image_dir, 'distance_mean_000.tiff'))
    video_kwargs = {
        'shape': depth_frame.shape[:2],
        'codec': 'h264',
        'fps': 30,
        'crf': 18
    }

    # depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    # p = 0.5
    # distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
    # lo, hi = [depth_curve_fn(x) for x in distance_limits]
    all_distances = []
    p = 0.5
    for idx in range(num):
        img = load_image(os.path.join(image_dir, f'distance_mean_{idx:03d}.tiff'))
        distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
        all_distances += distance_limits.tolist()
    lo, hi = np.min(all_distances), np.max(all_distances)
    # lo, hi = [depth_curve_fn(x) for x in [lo, hi]]

    for k in ['color', 'distance_mean']:
        video_file = os.path.join(out_dir, f'{out_name}_{k}.mp4')
        with media.VideoWriter(video_file, **video_kwargs, input_format='rgb') as writer:
            for idx in tqdm(range(num), desc=f'{out_name}_{k}'):
                if k == 'color':
                    img = load_image(os.path.join(image_dir, f'{k}_{idx:03d}.png'))
                    img = img / 255.
                elif k == 'distance_mean':
                    img = load_image(os.path.join(image_dir, f'{k}_{idx:03d}.tiff'))
                    # img = depth_curve_fn(img)
                    img = np.clip(1. - (img - lo) / (hi - lo), 0, 1)
                    img = cm.get_cmap('turbo')(img)[..., :3]
                img = (np.clip(np.nan_to_num(img), 0, 1) * 255).astype(np.uint8)
                writer.add_image(img)


image_root_dir = '/data/datasets/video_images'
video_dir = './video/'
os.makedirs(video_dir, exist_ok=True)
for method in os.listdir(image_root_dir):
    for scene_name in os.listdir(os.path.join(image_root_dir, method)):
        image_dir = os.path.join(image_root_dir, method, scene_name)
        out_name = f'{method}_{scene_name}'
        create_video(image_dir, video_dir, out_name)

