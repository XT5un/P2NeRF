# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
from os import path
import queue
import threading

from internal import math, utils  # pylint: disable=g-multiple-import
import jax
import numpy as np
from PIL import Image

import cv2


def load_dataset(split, train_dir, config):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  dataset_dict = {
      'replica_prior': ReplicaPrior,
  }
  return dataset_dict[config.dataset_loader](split, train_dir, config)


def convert_to_ndc(origins, directions, focal, width, height, near=1.,
                   focaly=None):
  """Convert a set of rays to normalized device coordinates (NDC).

  Args:
    origins: np.ndarray(float32), [..., 3], world space ray origins.
    directions: np.ndarray(float32), [..., 3], world space ray directions.
    focal: float, focal length.
    width: int, image width in pixels.
    height: int, image height in pixels.
    near: float, near plane along the negative z axis.
    focaly: float, Focal for y axis (if None, equal to focal).

  Returns:
    origins_ndc: np.ndarray(float32), [..., 3].
    directions_ndc: np.ndarray(float32), [..., 3].

  This function assumes input rays should be mapped into the NDC space for a
  perspective projection pinhole camera, with identity extrinsic matrix (pose)
  and intrinsic parameters defined by inputs focal, width, and height.

  The near value specifies the near plane of the frustum, and the far plane is
  assumed to be infinity.

  The ray bundle for the identity pose camera will be remapped to parallel rays
  within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
  world space can be remapped as long as it has dz < 0; this allows us to share
  a common NDC space for "forward facing" scenes.

  Note that
      projection(origins + t * directions)
  will NOT be equal to
      origins_ndc + t * directions_ndc
  and that the directions_ndc are not unit length. Rather, directions_ndc is
  defined such that the valid near and far planes in NDC will be 0 and 1.

  See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
  """

  # Shift ray origins to near plane, such that oz = -near.
  # This makes the new near bound equal to 0.
  t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions

  dx, dy, dz = np.moveaxis(directions, -1, 0)
  ox, oy, oz = np.moveaxis(origins, -1, 0)

  fx = focal
  fy = focaly if (focaly is not None) else focal

  # Perspective projection into NDC for the t = 0 near points
  #     origins + 0 * directions
  origins_ndc = np.stack([
      -2. * fx / width * ox / oz, -2. * fy / height * oy / oz,
      -np.ones_like(oz)
  ],
                         axis=-1)

  # Perspective projection into NDC for the t = infinity far points
  #     origins + infinity * directions
  infinity_ndc = np.stack([
      -2. * fx / width * dx / dz, -2. * fy / height * dy / dz,
      np.ones_like(oz)
  ],
                          axis=-1)

  # directions_ndc points from origins_ndc to infinity_ndc
  directions_ndc = infinity_ndc - origins_ndc

  return origins_ndc, directions_ndc


def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
  """Area downsample img (factor must evenly divide img height and width)."""
  sh = img.shape
  max_fn = lambda x: max(x, patch_size)
  out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
  img = cv2.resize(img, out_shape, mode)
  return img


def focus_pt_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]


def recenter_poses(poses):
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  return unpad_poses(poses)


def shift_origins(origins, directions, near=0.0):
  """Shift ray origins to near plane, such that oz = near."""
  t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions
  return origins


def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def generate_spiral_path(poses, bounds, n_frames=120, n_rots=2, zrate=.5):
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of near and far bounds in disparity space.
  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_poses.append(viewmatrix(z_axis, up, position))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def generate_spiral_path_dtu(poses, n_frames=120, n_rots=2, zrate=.5, perc=60):
  """Calculates a forward facing spiral path for rendering for DTU."""

  # Get radii for spiral path using 60th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), perc, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  z_axis = focus_pt_fn(poses)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    render_poses.append(viewmatrix(z_axis, up, position, True))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def generate_hemispherical_orbit(poses, n_frames=120):
  """Calculates a render path which orbits around the z-axis."""
  origins = poses[:, :3, 3]
  radius = np.sqrt(np.mean(np.sum(origins**2, axis=-1)))

  # Assume that z-axis points up towards approximate camera hemisphere
  sin_phi = np.mean(origins[:, 2], axis=0) / radius
  cos_phi = np.sqrt(1 - sin_phi**2)
  render_poses = []

  up = np.array([0., 0., 1.])
  for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
    camorigin = radius * np.array(
        [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi])
    render_poses.append(viewmatrix(camorigin, up, camorigin))

  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def transform_poses_to_hemisphere(poses, bounds):
  """Transforms input poses to lie roughly on the upper unit hemisphere."""

  # Use linear algebra to solve for the nearest point to the set of lines
  # given by each camera's focal axis
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]

  # Recenter poses around this point and such that the world space z-axis
  # points up toward the camera hemisphere (based on average camera origin)
  toward_cameras = origins[Ellipsis, 0].mean(0) - focus_pt
  arbitrary_dir = np.array([.1, .2, .3])
  cam2world = viewmatrix(toward_cameras, arbitrary_dir, focus_pt)
  poses_recentered = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  poses_recentered = poses_recentered[Ellipsis, :3, :4]

  # Rescale camera locations (and other metadata) such that average
  # squared distance to the origin is 1 (so cameras lie roughly unit sphere)
  origins = poses_recentered[:, :3, 3]
  avg_distance = np.sqrt(np.mean(np.sum(origins**2, axis=-1)))
  scale_factor = 1. / avg_distance
  poses_recentered[:, :3, 3] *= scale_factor
  bounds_recentered = bounds * scale_factor

  return poses_recentered, bounds_recentered


def subsample_patches(images, patch_size, batch_size, batching='all_images'):
  """Subsamples patches."""
  n_patches = batch_size // (patch_size ** 2)

  scale = np.random.randint(0, len(images))
  images = images[scale]

  if isinstance(images, np.ndarray):
    shape = images.shape
  else:
    shape = images.origins.shape

  # Sample images
  if batching == 'all_images':
    idx_img = np.random.randint(0, shape[0], size=(n_patches, 1))
  elif batching == 'single_image':
    idx_img = np.random.randint(0, shape[0])
    idx_img = np.full((n_patches, 1), idx_img, dtype=int)
  else:
    raise ValueError('Not supported batching type!')

  # Sample start locations
  x0 = np.random.randint(0, shape[2] - patch_size + 1, size=(n_patches, 1, 1))
  y0 = np.random.randint(0, shape[1] - patch_size + 1, size=(n_patches, 1, 1))
  xy0 = np.concatenate([x0, y0], axis=-1)
  patch_idx = xy0 + np.stack(
      np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
      axis=-1).reshape(1, -1, 2)

  # Subsample images
  if isinstance(images, np.ndarray):
    out = images[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
  else:
    out = utils.dataclass_map(
        lambda x: x[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(  # pylint: disable=g-long-lambda
            -1, x.shape[-1]), images)
  return out, np.ones((n_patches, 1), dtype=np.float32) * scale


def anneal_nearfar(d, it, near_final, far_final,
                   n_steps=2000, init_perc=0.2, mid_perc=0.5):
  """Anneals near and far plane."""
  mid = near_final + mid_perc * (far_final - near_final)

  near_init = mid + init_perc * (near_final - mid)
  far_init = mid + init_perc * (far_final - mid)

  weight = min(it * 1.0 / n_steps, 1.0)

  near_i = near_init + weight * (near_final - near_init)
  far_i = far_init + weight * (far_final - far_init)

  out_dict = {}
  for (k, v) in d.items():
    if 'rays' in k and isinstance(v, utils.Rays):
      ones = np.ones_like(v.origins[Ellipsis, :1])
      rays_out = utils.Rays(
          origins=v.origins, directions=v.directions,
          viewdirs=v.viewdirs, radii=v.radii,
          lossmult=v.lossmult, near=ones*near_i, far=ones*far_i)
      out_dict[k] = rays_out
    else:
      out_dict[k] = v
  return out_dict


def sample_recon_scale(image_list, dist='uniform_scale'):
  """Samples a scale factor for the reconstruction loss."""
  if dist == 'uniform_scale':
    idx = np.random.randint(len(image_list))
  elif dist == 'uniform_size':
    n_img = np.array([i.shape[0] for i in image_list], dtype=np.float32)
    probs = n_img / np.sum(n_img)
    idx = np.random.choice(np.arange(len(image_list)), size=(), p=probs)
  return idx


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, data_dir, config):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.use_tiffs = config.use_tiffs
    self.load_disps = config.compute_disp_metrics
    self.load_normals = config.compute_normal_metrics
    self.load_random_rays = config.load_random_rays
    self.load_random_fullimage_rays = config.dietnerf_loss_mult != 0.0
    self.load_masks = ((config.dataset_loader == 'dtu' or config.dataset_loader == 'dtu_prior') and (split == 'test')
                       and (not config.dtu_no_mask_eval)
                       and (not config.render_path))

    if split == 'train' and config.dataset_loader in ('replica_prior'):
      self.train_kpts_prior = config.kpts_loss_mult > 0.
      self.kpts_patch_size = config.kpts_patch_size
      self.kpts_init_iters = config.kpts_init_iters
      self.kpts_early_train_every = config.kpts_early_train_every
      self.train_depth_prior = config.depth_loss_mult > 0.
      self.depth_rank_level = config.depth_rank_level
      # self.use_sparsenerf_losses = False
      # if config.sparsenerf_loss_mult > 0.:
      #   self.use_sparsenerf_losses = True
      #   self.train_depth_prior = False
      # self.use_colmap_warmup = False
      # if config.use_colmap_warmup and config.dataset_loader == 'replica_prior' and config.replica_scene in (
      #   'scene0710_00', 'scene_0758_00', 'scene0781_00'):
      #   self.use_colmap_warmup = True

    self.split = split
    if config.dataset_loader == 'replica_prior':
      self.data_base_dir = data_dir
      data_dir = os.path.join(data_dir, config.replica_scene)
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    self.near_origin = config.near_origin
    self.anneal_nearfar = config.anneal_nearfar
    self.anneal_nearfar_steps = config.anneal_nearfar_steps
    self.anneal_nearfar_perc = config.anneal_nearfar_perc
    self.anneal_mid_perc = config.anneal_mid_perc
    self.sample_reconscale_dist = config.sample_reconscale_dist

    if split == 'train':
      self._train_init(config)
    elif split == 'test':
      self._test_init(config)
    else:
      raise ValueError(
          f'`split` should be \'train\' or \'test\', but is \'{split}\'.')
    self.batch_size = config.batch_size // jax.host_count()
    self.batch_size_random = config.batch_size_random // jax.host_count()
    print('Using following batch size', self.batch_size)
    self.patch_size = config.patch_size
    self.batching = config.batching
    self.batching_random = config.batching_random
    self.render_path = config.render_path
    self.render_train = config.render_train
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self.queue.get()
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    if self.split == 'train':
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    self._generate_downsampled_images(config)
    self._generate_rays(config)
    self._generate_downsampled_rays(config)
    # Generate more rays / image patches for unobserved-view-based losses.
    if self.load_random_rays:
      self._generate_random_rays(config)
    if self.load_random_fullimage_rays:
      self._generate_random_fullimage_rays(config)
      self._load_renderings_featloss(config)

    self.it = 0
    self.images_noreshape = self.images[0]

    if config.batching == 'all_images':
      # flatten the ray and image dimension together.
      self.images = [i.reshape(-1, 3) for i in self.images]
      if self.load_disps:
        self.disp_images = self.disp_images.flatten()
      if self.load_normals:
        self.normal_images = self.normal_images.reshape([-1, 3])

      self.ray_noreshape = [self.rays]
      self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
          [-1, r.shape[-1]]), i) for (i, res) in zip(
              self.rays, self.resolutions)]

    elif config.batching == 'single_image':
      self.images = [i.reshape(
          [-1, r, 3]) for (i, r) in zip(self.images, self.resolutions)]
      if self.load_disps:
        self.disp_images = self.disp_images.reshape([-1, self.resolution])
      if self.load_normals:
        self.normal_images = self.normal_images.reshape(
            [-1, self.resolution, 3])

      self.ray_noreshape = [self.rays]
      self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
          [-1, res, r.shape[-1]]), i) for (i, res) in  # pylint: disable=cell-var-from-loop
                   zip(self.rays, self.resolutions)]
    else:
      raise NotImplementedError(
          f'{config.batching} batching strategy is not implemented.')

  def _test_init(self, config):
    self._load_renderings(config)
    if self.load_masks:
      self._load_masks(config)
    self._generate_rays(config)
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""

    self.it = self.it + 1

    return_dict = {}
    if self.batching == 'all_images':
      # sample scale
      idxs = sample_recon_scale(self.images, self.sample_reconscale_dist)
      ray_indices = np.random.randint(0, self.rays[idxs].origins.shape[0],
                                      (self.batch_size,))
      return_dict['rgb'] = self.images[idxs][ray_indices]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[ray_indices],
                                                self.rays[idxs])
      if self.load_disps:
        return_dict['disps'] = self.disp_images[ray_indices]
      if self.load_normals:
        return_dict['normals'] = self.normal_images[ray_indices]
    elif self.batching == 'single_image':
      idxs = sample_recon_scale(self.images, self.sample_reconscale_dist)
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays[idxs].origins[0].shape[0],
                                      (self.batch_size,))
      return_dict['rgb'] = self.images[idxs][image_index][ray_indices]
      return_dict['rays'] = utils.dataclass_map(
          lambda r: r[image_index][ray_indices], self.rays[idxs])
      if self.load_disps:
        return_dict['disps'] = self.disp_images[image_index][ray_indices]
      if self.load_normals:
        return_dict['normals'] = self.normal_images[image_index][ray_indices]
    else:
      raise NotImplementedError(
          f'{self.batching} batching strategy is not implemented.')

    if self.load_random_rays:
      return_dict['rays_random'], return_dict['rays_random_scale'] = (
          subsample_patches(self.random_rays, self.patch_size,
                            self.batch_size_random,
                            batching=self.batching_random))
      return_dict['rays_random2'], return_dict['rays_random2_scale'] = (
          subsample_patches(
              self.random_rays, self.patch_size, self.batch_size_random,
              batching=self.batching_random))
    if self.load_random_fullimage_rays:
      idx_img = np.random.randint(self.random_fullimage_rays.origins.shape[0])
      return_dict['rays_feat'] = utils.dataclass_map(
          lambda x: x[idx_img].reshape(-1, x.shape[-1]),
          self.random_fullimage_rays)
      idx_img = np.random.randint(self.images_feat.shape[0])
      return_dict['image_feat'] = self.images_feat[idx_img].reshape(-1, 3)

    if self.anneal_nearfar:
      return_dict = anneal_nearfar(return_dict, self.it, self.near, self.far,
                                   self.anneal_nearfar_steps,
                                   self.anneal_nearfar_perc,
                                   self.anneal_mid_perc)
    return return_dict

  def _next_test(self):
    """Sample next test example."""

    return_dict = {}

    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      return_dict['rays'] = utils.dataclass_map(lambda r: r[idx],
                                                self.render_rays)
    else:
      return_dict['rgb'] = self.images[idx]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[idx], self.rays)

    if self.load_masks:
      return_dict['mask'] = self.masks[idx]
    if self.load_disps:
      return_dict['disps'] = self.disp_images[idx]
    if self.load_normals:
      return_dict['normals'] = self.normal_images[idx]

    return return_dict

  def _generate_rays(self, config):
    """Generating rays for all images."""
    del config  # Unused.
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.width, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.height, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_dirs = np.stack(
        [(x - self.width * 0.5 + 0.5) / self.focal,
         -(y - self.height * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
        axis=-1)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                   self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see paper).
    radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[Ellipsis, :1])
    self.rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        lossmult=ones,
        radii=radii,
        near=ones * self.near,
        far=ones * self.far)
    self.render_rays = self.rays

  def _generate_random_poses(self, config):
    """Generates random poses."""
    if config.random_pose_type == 'allposes':
      random_poses = list(self.camtoworlds_all)
    elif config.random_pose_type == 'renderpath':
      def sample_on_sphere(n_samples, only_upper=True, radius=4.03112885717555):
        p = np.random.randn(n_samples, 3)
        if only_upper:
          p[:, -1] = abs(p[:, -1])
        p = p / np.linalg.norm(p, axis=-1, keepdims=True) * radius
        return p

      def create_look_at(eye, target=np.array([0, 0, 0]),
                         up=np.array([0, 0, 1]), dtype=np.float32):
        """Creates lookat matrix."""
        eye = eye.reshape(-1, 3).astype(dtype)
        target = target.reshape(-1, 3).astype(dtype)
        up = up.reshape(-1, 3).astype(dtype)

        def normalize_vec(x, eps=1e-9):
          return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

        forward = normalize_vec(target - eye)
        side = normalize_vec(np.cross(forward, up))
        up = normalize_vec(np.cross(side, forward))

        up = up * np.array([1., 1., 1.]).reshape(-1, 3)
        forward = forward * np.array([-1., -1., -1.]).reshape(-1, 3)

        rot = np.stack([side, up, forward], axis=-1).astype(dtype)
        return rot

      origins = sample_on_sphere(config.n_random_poses)
      rotations = create_look_at(origins)
      random_poses = np.concatenate([rotations, origins[:, :, None]], axis=-1)
    else:
      raise ValueError('Not supported random pose type.')
    self.random_poses = np.stack(random_poses, axis=0)

  def _generate_random_rays(self, config):
    """Generating rays for all images."""
    self._generate_random_poses(config)

    random_rays = []
    for sfactor in [2**i for i in range(config.random_scales_init,
                                        config.random_scales)]:
      w = self.width // sfactor
      h = self.height // sfactor
      f = self.focal / (sfactor * 1.0)
      x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32),  # X-Axis (columns)
          np.arange(h, dtype=np.float32),  # Y-Axis (rows)
          indexing='xy')
      camera_dirs = np.stack(
          [(x - w * 0.5 + 0.5) / f,
           -(y - h * 0.5 + 0.5) / f, -np.ones_like(x)],
          axis=-1)
      directions = ((camera_dirs[None, Ellipsis, None, :] *
                     self.random_poses[:, None, None, :3, :3]).sum(axis=-1))
      origins = np.broadcast_to(self.random_poses[:, None, None, :3, -1],
                                directions.shape)
      viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.sqrt(
          np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
      dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
      # Cut the distance in half, multiply it to match the variance of a uniform
      # distribution the size of a pixel (1/12, see paper).
      radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

      ones = np.ones_like(origins[Ellipsis, :1])
      rays = utils.Rays(
          origins=origins,
          directions=directions,
          viewdirs=viewdirs,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far)
      random_rays.append(rays)
    self.random_rays = random_rays

  def _load_renderings_featloss(self, config):
    """Loades renderings for DietNeRF's feature loss."""
    images = self.images[0]
    res = config.dietnerf_loss_resolution
    images_feat = []
    for img in images:
      images_feat.append(cv2.resize(img, (res, res), cv2.INTER_AREA))
    self.images_feat = np.stack(images_feat)

  def _generate_random_fullimage_rays(self, config):
    """Generating random rays for full images."""
    self._generate_random_poses(config)

    width = config.dietnerf_loss_resolution
    height = config.dietnerf_loss_resolution
    f = self.focal / (self.width * 1.0 / width)

    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(width, dtype=np.float32) + .5,
        np.arange(height, dtype=np.float32) + .5,
        indexing='xy')

    camera_dirs = np.stack([(x - width * 0.5 + 0.5) / f,
                            -(y - height * 0.5 + 0.5) / f,
                            -np.ones_like(x)], axis=-1)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                   self.random_poses[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.random_poses[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see paper).
    radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[Ellipsis, :1])
    self.random_fullimage_rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)

  def _generate_downsampled_images(self, config):
    """Generating downsampled images."""
    images = []
    resolutions = []
    for sfactor in [2**i for i in range(config.recon_loss_scales)]:
      imgi = np.stack([downsample(i, sfactor) for i in self.images])
      images.append(imgi)
      resolutions.append(imgi.shape[1] * imgi.shape[2])

    self.images = images
    self.resolutions = resolutions

  def _generate_downsampled_rays(self, config):
    """Generating downsampled images."""
    rays, height, width, focal = self.rays, self.height, self.width, self.focal
    ray_list = [rays]
    for sfactor in [2**i for i in range(1, config.recon_loss_scales)]:
      self.height = height // sfactor
      self.width = width // sfactor
      self.focal = focal * 1.0 / sfactor
      self._generate_rays(config)
      ray_list.append(self.rays)
    self.height = height
    self.width = width
    self.focal = focal
    self.rays = ray_list

##### Prior Dataset #####

def get_kpts_batch(kpts_list, patch_size, batch_size, width, height):
  image_index = np.random.randint(len(kpts_list))
  kpts = kpts_list[image_index]['kpts']
  depth = kpts_list[image_index]['depth']

  kpts_num = batch_size // (patch_size ** 2)
  rand_idxs = np.random.choice(len(kpts), kpts_num)
  kpts = kpts[rand_idxs]
  depth = depth[rand_idxs]

  patch_coords = np.meshgrid(np.arange(-patch_size // 2, patch_size // 2),
                             np.arange(-patch_size // 2, patch_size // 2),
                             indexing='xy')
  patch_coords = np.stack(patch_coords, axis=-1)
  patch_coords = kpts[:, None, None, :] + patch_coords[None, ...] # (kpts_num, patch_size, patch_size, 2)
  patch_coords[..., 0] = np.clip(patch_coords[..., 0], 0, width - 1)
  patch_coords[..., 1] = np.clip(patch_coords[..., 1], 0, height - 1)

  patch_depth = np.broadcast_to(depth[:, None, None], patch_coords.shape[:-1]).reshape(-1)
  ray_indices = (patch_coords[..., 1] * width + patch_coords[..., 0]).reshape(-1)

  return image_index, ray_indices, patch_depth

def get_kpts_pair_batch(kpts_dict, patch_size, batch_size, width, height):
  pair_name = list(kpts_dict.keys())[np.random.randint(len(kpts_dict))]
  image_index0, image_index1 = pair_name
  kpts0, kpts1 = kpts_dict[pair_name]['kpts0'], kpts_dict[pair_name]['kpts1']
  depth0, depth1 = kpts_dict[pair_name]['depth0'], kpts_dict[pair_name]['depth1']

  kpts_num = batch_size // 2 // (patch_size ** 2)
  rand_idxs = np.random.permutation(len(kpts0))[:kpts_num]
  if len(rand_idxs) < kpts_num:
    rand_idxs = np.concatenate([rand_idxs, np.random.randint(len(kpts0), size=kpts_num - len(rand_idxs))])
  kpts0 = kpts0[rand_idxs]
  kpts1 = kpts1[rand_idxs]
  depth0 = depth0[rand_idxs]
  depth1 = depth1[rand_idxs]

  patch_coords = np.meshgrid(np.arange(-patch_size // 2, patch_size // 2),
                             np.arange(-patch_size // 2, patch_size // 2),
                             indexing='xy')
  patch_coords = np.stack(patch_coords, axis=-1)
  patch_coords0 = kpts0[:, None, None, :] + patch_coords[None, ...] # (kpts_num, patch_size, patch_size, 2)
  patch_coords1 = kpts1[:, None, None, :] + patch_coords[None, ...]
  patch_coords0[..., 0] = np.clip(patch_coords0[..., 0], 0, width - 1)
  patch_coords0[..., 1] = np.clip(patch_coords0[..., 1], 0, height - 1)
  patch_coords1[..., 0] = np.clip(patch_coords1[..., 0], 0, width - 1)
  patch_coords1[..., 1] = np.clip(patch_coords1[..., 1], 0, height - 1)

  patch_depth0 = np.broadcast_to(depth0[:, None, None], patch_coords0.shape[:-1]).reshape(-1)
  patch_depth1 = np.broadcast_to(depth1[:, None, None], patch_coords1.shape[:-1]).reshape(-1)
  ray_indices0 = (patch_coords0[..., 1] * width + patch_coords0[..., 0]).reshape(-1)
  ray_indices1 = (patch_coords1[..., 1] * width + patch_coords1[..., 0]).reshape(-1)

  return image_index0, image_index1, ray_indices0, ray_indices1, patch_depth0, patch_depth1

def get_depth_rank_batch(image_num, mono_depths, batch_size, width, height, rank_level):
  # rank_level = 32
  group_num = batch_size // rank_level
  image_index = np.random.randint(image_num)
  depth = mono_depths[image_index]
  depth = depth.reshape(-1)

  # calculate rank map
  q = [i * 100.0 / rank_level for i in range(1, rank_level)]
  vmin = np.percentile(depth, 1)
  vmax = np.percentile(depth, 99)
  th = np.percentile(depth[(depth > vmin) & (depth < vmax)], q)
  rank_map = np.zeros(depth.shape, dtype=np.int32)
  for i in range(len(th)):
      rank_map[depth > th[i]] = i + 1
  
  # group sample
  ray_indices = []
  for i in range(rank_level):
    idxs = np.where(rank_map == i)[0]
    rand_idxs = np.random.permutation(len(idxs))[:group_num]
    if rand_idxs.shape[0] < group_num:
      rand_idxs = np.concatenate([rand_idxs, np.random.randint(len(idxs), size=group_num - rand_idxs.shape[0])])
    ray_indices.append(idxs[rand_idxs])
  ray_indices = np.stack(ray_indices, axis=0).transpose() # (rank_level, group_num)
  ray_indices = ray_indices.reshape(-1)
  return image_index, ray_indices


def load_kpts_prior(prior_dir, image_names, factor, width, height):
  kpts_dict = {}
  kpts_list = []
  for _ in range(len(image_names)):
    kpts_list.append({'depth': [], 'kpts': []})

  for pair_name in os.listdir(prior_dir):
    data = np.load(path.join(prior_dir, pair_name))
    name0 = data['name0'][()]
    name1 = data['name1'][()]
    kpts0 = data['kpts0']
    kpts1 = data['kpts1']
    img_idx0 = image_names.index(name0)
    img_idx1 = image_names.index(name1)
    if factor > 1:
      kpts0 = (kpts0 / factor).astype(np.int32)
      kpts1 = (kpts1 / factor).astype(np.int32)
    depth0 = data['depth0']
    depth1 = data['depth1']

    kpts_dict[(img_idx0, img_idx1)] = {
      'kpts0': kpts0,
      'kpts1': kpts1,
      'depth0': depth0,
      'depth1': depth1,
    }
    kpts_list[img_idx0]['kpts'].append(kpts0)
    kpts_list[img_idx1]['kpts'].append(kpts1)
    kpts_list[img_idx0]['depth'].append(depth0)
    kpts_list[img_idx1]['depth'].append(depth1)
  
  for i in range(len(kpts_list)):
    kpts_list[i]['kpts'] = np.concatenate(kpts_list[i]['kpts'], axis=0)
    kpts_list[i]['depth'] = np.concatenate(kpts_list[i]['depth'], axis=0)

  return kpts_dict, kpts_list

def load_depth_prior(prior_dir, filenames, factor):
  mono_depths = []
  for fname in filenames:
    frame_path = path.join(prior_dir, fname)
    depth_prior = cv2.imread(frame_path, -1)
    if factor > 1:
      out_shape = (depth_prior.shape[1] // factor, depth_prior.shape[0] // factor)
      depth_prior = cv2.resize(depth_prior, out_shape, interpolation=cv2.INTER_AREA)
    depth_prior = depth_prior.astype(np.float32)
    mono_depths.append(depth_prior)
  return np.stack(mono_depths, axis=0)

class ReplicaPrior(Dataset):
  def _load_renderings(self, config):

    task = 'video' if config.render_path else self.split
    with open(path.join(self.data_dir, f'transforms_{task}.json'), 'r') as f:
      meta = json.load(f)

    self.near = meta['near']
    self.far = meta['far']
    self.focal = meta['frames'][0]['fx']
    if config.factor > 1:
      self.focal /= config.factor
    images = []
    cams = []
    image_names = []

    for frame in meta['frames']:
      cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
      if task != 'video':
        image = cv2.imread(path.join(self.data_dir, frame['file_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if config.factor > 1:
          out_shape = (image.shape[1] // config.factor, image.shape[0] // config.factor)
          image = cv2.resize(image, out_shape, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.
        name = os.path.splitext(os.path.basename(frame['file_path']))[0]

        images.append(image)
        image_names.append(name)

    dataset_name = 'replica'
    if config.replica_scene in ('office0', 'office1', 'office2', 'office3'
                                'office4', 'room0', 'room1', 'room2'):
      dataset_name = 'replica'
    elif config.replica_scene in ('scene0710_00', 'scene0758_00', 'scene0781_00'):
      dataset_name = 'DDP'

    self.camtoworlds_all = np.stack(cams, axis=0)
    self.camtoworlds = self.camtoworlds_all.copy()
    if task == 'video':
      self.height, self.width = (680, 1200) if dataset_name == 'replica' else (468, 624)
      self.height = self.height // config.factor
      self.width = self.width // config.factor
      self.n_examples = self.camtoworlds.shape[0]
    else:
      self.images_all = np.stack(images, axis=0)
      self.height, self.width = self.images_all.shape[1:3]
      self.images = self.images_all.copy()
      self.n_examples = self.images.shape[0]
    self.resolution = self.height * self.width

    if task == 'train':
      depth_prior_dir = path.join(config.prior_dir, f'{dataset_name}/{config.replica_scene}/train/depth')
      self.mono_depths = load_depth_prior(depth_prior_dir, [_ + '.png' for _ in image_names], config.factor)

      kpts_prior_dir = path.join(config.prior_dir, f'{dataset_name}/{config.replica_scene}/train/keypoints_pair')
      kpts_dict, kpts_list = load_kpts_prior(kpts_prior_dir, image_names, config.factor, self.width, self.height)
      self.kpts_dict = kpts_dict
      self.kpts_list = kpts_list
      

  def _generate_random_poses(self, config):
    # raw camera bbox
    raw_camera_positions = self.camtoworlds_all[:, :3, 3]
    bbox_min = raw_camera_positions.min(axis=0)
    bbox_max = raw_camera_positions.max(axis=0)
    bbox_size = (bbox_max - bbox_min) * 1.1
    bbox_center = bbox_min + bbox_size / 2
    bbox_min = bbox_center - bbox_size / 2
    bbox_max = bbox_center + bbox_size / 2

    # generate random positions
    random_positions = np.random.uniform(bbox_min, bbox_max, size=(config.n_random_poses, 3))

    # find nearest camera
    dists = np.linalg.norm(random_positions[:, None] - raw_camera_positions[None, :], axis=-1)
    nearest_camera_idxs = dists.argmin(axis=-1)
    random_rotate = self.camtoworlds_all[nearest_camera_idxs][:, :3, :3]

    random_poses = np.concatenate([random_rotate, random_positions[:, :, None]], axis=-1)
    self.random_poses = random_poses

  def _train_init(self, config):
    super()._train_init(config)

    self.images = self.images[0]
    self.rays = self.rays[0]

  def _next_train(self):
    
    self.it = self.it + 1
    return_dict = {}

    is_kpts_iter = False
    if self.train_kpts_prior:
      if self.it < self.kpts_init_iters * self.kpts_early_train_every and \
        self.it % self.kpts_early_train_every == 0:
        is_kpts_iter = True

    ### 不要用多GPU跑，否则这里会出错 !!!!!!!!
    if is_kpts_iter:
      (
        image_index0, image_index1, ray_indices0, ray_indices1, patch_depth0, patch_depth1
      ) = get_kpts_pair_batch(self.kpts_dict, self.kpts_patch_size, self.batch_size, self.width, self.height)
      return_dict['rgb'] = np.concatenate([self.images[image_index0][ray_indices0], self.images[image_index1][ray_indices1]], axis=0)
      return_dict['rays'] = utils.dataclass_map(
        lambda r: np.concatenate([r[image_index0][ray_indices0], r[image_index1][ray_indices1]], axis=0), self.rays)
      return_dict['kpts_depth'] = np.concatenate([patch_depth0, patch_depth1], axis=0)
    else:
      image_index, ray_indices = get_depth_rank_batch(
        len(self.images), self.mono_depths, self.batch_size, self.width, self.height, self.depth_rank_level)
      return_dict['rgb'] = self.images[image_index][ray_indices]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[image_index][ray_indices], self.rays)
    
    if self.load_random_rays:
      return_dict['rays_random'], return_dict['rays_random_scale'] = (
          subsample_patches(self.random_rays, self.patch_size,
                            self.batch_size_random,
                            batching=self.batching_random))
      return_dict['rays_random2'], return_dict['rays_random2_scale'] = (
          subsample_patches(
              self.random_rays, self.patch_size, self.batch_size_random,
              batching=self.batching_random))
    
    if self.anneal_nearfar:
      return_dict = anneal_nearfar(return_dict, self.it, self.near, self.far,
                                   self.anneal_nearfar_steps,
                                   self.anneal_nearfar_perc,
                                   self.anneal_mid_perc)
    
    return return_dict