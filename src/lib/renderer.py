from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    camera_position_from_spherical_angles,
)
from pytorch3d.renderer.mesh import utils as p3d_utils
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes

from src.lib.mesh_utils import (
    campos_to_R_T,
    pre_process_mesh_pascal,
    set_bary_coords_to_nearest,
)


@dataclass
class RenderCfg:
    image_shape: ListConfig[int]
    downsample_rate: int
    max_n: int
    viewport: int
    focal_length: float


class RenderEngine:
    cfg: RenderCfg
    rasterizer: MeshRasterizer

    def __init__(self, cfg: DictConfig):
        render_cfg = deepcopy(cfg.dataset)
        with open_dict(render_cfg):
            render_cfg.max_n = cfg.model.mesh.max_n
            render_cfg.downsample_rate = cfg.model.extractor.downsample_rate
        self.cfg = render_cfg
        map_shape = (
            self.cfg.image_shape[0] // self.cfg.downsample_rate,
            self.cfg.image_shape[1] // self.cfg.downsample_rate,
        )
        M = self.cfg.viewport * self.cfg.focal_length
        cameras = PerspectiveCameras(
            focal_length=M // self.cfg.downsample_rate,
            principal_point=((map_shape[1] // 2, map_shape[0] // 2),),
            image_size=(map_shape,),
            in_ndc=False,
        ).cuda()
        raster_settings = RasterizationSettings(
            image_size=map_shape,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

    def get_annotations(self, texture, pose_gd, labels, xverts, xfaces, device):
        dist, elev, azim, theta = (
            pose_gd[:, 0],
            pose_gd[:, 1],
            pose_gd[:, 2],
            pose_gd[:, 3],
        )

        C = camera_position_from_spherical_angles(dist, elev, azim, degrees=False)
        texture = [texture[label] for label in labels]
        xverts = [xverts[label] for label in labels]
        xfaces = [xfaces[label] for label in labels]
        inter_module = MeshInterpolateModule(
            xverts,
            xfaces,
            texture,
            self.rasterizer,
        )

        inter_module = inter_module.to(device)
        projected_map, v_vis = inter_module(C.to(device), theta, get_visibility=True)

        kp = inter_module.rasterizer.cameras.transform_points(
            inter_module.meshes.verts_padded()
        )
        kp = self.rotate_keypoints(
            (
                self.cfg.image_shape[1] // (2 * self.cfg.downsample_rate),
                self.cfg.image_shape[0] // (2 * self.cfg.downsample_rate),
            ),
            kp,
            180,
        )
        kp = torch.stack(
            [
                torch.cat(
                    [
                        kp[i],
                        -torch.ones(
                            (self.cfg.max_n - kp[i].shape[0], 3), device=device
                        ),
                    ]
                )
                for i in range(len(kp))
            ],
            dim=0,
        )
        kp_vis = torch.zeros((projected_map.shape[0], self.cfg.max_n), device=device)
        obj_mask = torch.zeros_like(projected_map[:, 0, ...], device=device)
        for i, label in enumerate(labels):
            current_verts = xverts[i].shape[0]
            vis = v_vis[0:current_verts]
            v_vis = v_vis[current_verts:]
            kp_vis[i, 0:current_verts] = vis
            obj_mask[i, ...] = torch.any(projected_map[i] != 0.0, dim=0).float()
        kp = kp[..., [1, 0]]

        return kp, kp_vis, obj_mask, projected_map

    def render(self, texture, C, theta, labels, xverts, xfaces, device, box=True):
        texture = [texture[label] for label in labels]
        xverts = [xverts[label] for label in labels]
        xfaces = [xfaces[label] for label in labels]
        inter_module = MeshInterpolateModule(
            xverts,
            xfaces,
            texture,
            self.rasterizer,
        )

        inter_module = inter_module.to(device)
        projected_map, v_vis = inter_module(C.to(device), theta, get_visibility=True)

        if not box:
            return projected_map
        obj_mask = torch.zeros_like(projected_map[:, 0, ...], device=device)
        for i, label in enumerate(labels):
            obj_mask[i, ...] = torch.any(projected_map[i] != 0.0, dim=0).float()
        return obj_mask, projected_map

    def rotate_keypoints(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """

        angle = torch.deg2rad(
            torch.ones(point.shape[0], device=point.device) * angle
        ).unsqueeze(-1)
        ox, oy = origin
        px, py = point[..., 0], point[..., 1]

        qx = ox + torch.cos(angle) * (px - ox) - torch.sin(angle) * (py - oy)
        qy = oy + torch.sin(angle) * (px - ox) + torch.cos(angle) * (py - oy)
        return torch.stack((qx, qy, point[..., 2]), dim=-1)


class MeshInterpolateModule(nn.Module):
    def __init__(
        self,
        vertices,
        faces,
        memory_bank,
        rasterizer,
        post_process=None,
        off_set_mesh=False,
    ):
        super().__init__()

        # Convert memory features of vertices to faces
        self.face_memory = None
        self._update_memory(memory_bank=memory_bank, faces=faces)

        # Support multiple meshes at same time
        if type(vertices) == list:
            self.n_mesh = len(vertices)
            # Preprocess convert mesh in PASCAL3d+ standard to Pytorch3D
            verts = [pre_process_mesh_pascal(t) for t in vertices]

            # Create Pytorch3D meshes
            self.meshes = Meshes(verts=verts, faces=faces, textures=None)
        else:
            self.n_mesh = 1
            # Preprocess convert meshes in PASCAL3d+ standard to Pytorch3D
            verts = pre_process_mesh_pascal(vertices)

            # Create Pytorch3D meshes
            self.meshes = Meshes(verts=[verts], faces=[faces], textures=None)

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def _update_memory(self, memory_bank, faces=None):
        if type(memory_bank) == list:
            if faces is None:
                faces = self.faces
            # Convert memory features of vertices to faces
            self.face_memory = torch.cat(
                [m[f.type(torch.long)] for m, f in zip(memory_bank, faces)],
                dim=0,
            )
        else:
            if faces is None:
                faces = self.faces
            # Convert memory features of vertices to faces
            self.face_memory = memory_bank[faces.type(torch.long)]

    def to(self, *args, **kwargs):
        if "device" in kwargs.keys():
            device = kwargs["device"]
        else:
            device = args[0]
        super().to(device)

        self.meshes = self.meshes.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(
        self,
        campos,
        theta,
        blur_radius=0,
        deform_verts=None,
        mode="bilinear",
        get_visibility=False,
        **kwargs,
    ):
        R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)

        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes

        n_cam = campos.shape[0]
        if n_cam > 1 and self.n_mesh > 1:
            get = self._forward_interpolate(
                R,
                T,
                meshes,
                self.face_memory,
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
                get_visibility=get_visibility,
            )
        elif n_cam > 1 and self.n_mesh == 1:
            get = self._forward_interpolate(
                R,
                T,
                meshes.extend(campos.shape[0]),
                self.face_memory.repeat(campos.shape[0], 1, 1).view(
                    -1,
                    *self.face_memory.shape[1:],
                ),
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
                get_visibility=get_visibility,
            )
        else:
            get = self._forward_interpolate(
                R,
                T,
                meshes,
                self.face_memory,
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
                get_visibility=get_visibility,
            )
        if get_visibility:
            get, kp_vis = get
            return get, kp_vis

        return get

    def _forward_interpolate(
        self,
        R,
        T,
        meshes,
        face_memory,
        rasterizer,
        blur_radius=0,
        mode="bilinear",
        get_visibility=False,
    ):
        fragments = self._rasterize(R, T, meshes, rasterizer, blur_radius=blur_radius)

        if mode == "nearest":
            out_map = p3d_utils.interpolate_face_attributes(
                fragments.pix_to_face,
                set_bary_coords_to_nearest(fragments.bary_coords),
                face_memory,
            )
        else:
            out_map = p3d_utils.interpolate_face_attributes(
                fragments.pix_to_face,
                fragments.bary_coords,
                face_memory,
            )

        out_map = out_map.squeeze(dim=3)
        out_map = out_map.transpose(3, 2).transpose(2, 1)
        if not get_visibility:
            return out_map

        pix_to_face = fragments.pix_to_face
        packed_faces = meshes.faces_packed()
        packed_verts = meshes.verts_packed()

        vertex_visibility_map = torch.zeros(packed_verts.shape[0])
        visible_faces = pix_to_face.unique()
        visible_faces = visible_faces[1:] if visible_faces[0] == -1 else visible_faces
        visible_verts_idx = packed_faces[visible_faces]
        unique_visible_verts_idx = torch.unique(visible_verts_idx)
        vertex_visibility_map[unique_visible_verts_idx] = 1.0

        return out_map, vertex_visibility_map

    def _rasterize(self, R, T, meshes, rasterizer, blur_radius=0):
        # It will automatically update the camera settings -> R, T in rasterizer.camera
        fragments = rasterizer(meshes, R=R, T=T)

        if blur_radius > 0.0:
            clipped_bary_coords = p3d_utils._clip_barycentric_coordinates(
                fragments.bary_coords
            )
            clipped_zbuf = p3d_utils._interpolate_zbuf(
                fragments.pix_to_face,
                clipped_bary_coords,
                meshes,
            )
            fragments = Fragments(
                bary_coords=clipped_bary_coords,
                zbuf=clipped_zbuf,
                dists=fragments.dists,
                pix_to_face=fragments.pix_to_face,
            )
        return fragments
