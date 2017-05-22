#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pykitti

from camera_matrix.camera import recover_intrinsic_extrinsic, backproject, project
from camera_matrix.utils import rot_matrix, yaw_pitch_roll_from_rot_matrix
from camera_matrix.plot_utils import Arrow3D, Frustum, create_slider


class PinholeCameraVisualization(object):
    def __init__(self, img_w, img_h, camera_matrix_file, plane_normal, plane_point,
                 world_boundary=2):
        super(PinholeCameraVisualization, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.plane_normal = plane_normal
        self.plane_point = plane_point

        fig = plt.figure()
        plt.subplots_adjust(bottom=0.2)
        ax_image = fig.add_subplot(221)
        ax_world = fig.add_subplot(222, projection='3d')
        ax_image.set_xlim(0, img_w)
        ax_image.set_ylim(0, img_h)
        ax_world.set_xlim(-world_boundary, world_boundary)
        ax_world.set_ylim(-world_boundary, world_boundary)
        ax_world.set_zlim(-world_boundary, world_boundary)
        ax_world.set_xlabel('x')
        ax_world.set_ylabel('y')
        ax_world.set_zlabel('z')


        s_x0 = create_slider([0.15, 0.47, 0.65, 0.03], 'x0', -img_w // 2, img_w, img_w // 2)
        s_y0 = create_slider([0.15, 0.42, 0.65, 0.03], 'y0', -img_h // 2, img_h, img_h // 2)
        s_f  = create_slider([0.15, 0.37, 0.65, 0.03], 'f', 1, 3000., 450.)
        s_s  = create_slider([0.15, 0.32, 0.65, 0.03], 's', -img_h // 2, img_h // 2, 0.)
        s_tx = create_slider([0.15, 0.27, 0.65, 0.03], 'tx', -world_boundary, world_boundary, 0.)
        s_ty = create_slider([0.15, 0.22, 0.65, 0.03], 'ty', -world_boundary, world_boundary, 0.)
        s_tz = create_slider([0.15, 0.17, 0.65, 0.03], 'tz', -world_boundary, world_boundary, -2.)
        s_rx = create_slider([0.15, 0.12, 0.65, 0.03], 'rx', -270, 270, 0.)
        s_ry = create_slider([0.15, 0.07, 0.65, 0.03], 'ry', -270, 270, 0.)
        s_rz = create_slider([0.15, 0.02, 0.65, 0.03], 'rz', -270, 270, 0.)

        self.l_world = None

        if camera_matrix_file is not None:
            filedata = pykitti.utils.read_calib_file(camera_matrix_file)
            camera_matrix = np.reshape(filedata['P2'], (3, 4))
            print("Loaded camera matrix:")
            print(camera_matrix)
            K, E = recover_intrinsic_extrinsic(camera_matrix)
            print("Decomposed K:")
            print(K)
            print("Decomposed E:")
            print(E)
            s_x0.set_val(K[0, 2])
            s_y0.set_val(K[1, 2])
            s_f.set_val(K[0, 0])
            s_s.set_val(K[0, 1])
            t = E[:, 3]
            R = E[:, :3]
            cam_pos = -R.T.dot(t)
            print("cam_pos: {}".format(cam_pos))
            s_tx.set_val(cam_pos[0])
            s_ty.set_val(cam_pos[1])
            s_tz.set_val(cam_pos[2])
            yaw, pitch, roll = yaw_pitch_roll_from_rot_matrix(R.T)
            print("yaw, roll, pitch: {} {} {} deg".format(np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)))
            s_rx.set_val(pitch)
            s_ry.set_val(yaw)
            s_rz.set_val(roll)
            print(" -- camera_matrix loading complete -- ")

        def update(val):
            x0 = s_x0.val
            y0 = s_y0.val
            fx = s_f.val
            fy = s_f.val
            s = s_s.val
            cam_pos = np.array([s_tx.val, s_ty.val, s_tz.val])
            R_cam = rot_matrix(np.deg2rad(s_rx.val),
                               np.deg2rad(s_ry.val),
                               np.deg2rad(s_rz.val))

            fov_h = 2*np.arctan(img_w / (2*fx))
            fov_v = 2*np.arctan(img_h / (2*fy))
            print("hor. FoV: {}".format(np.rad2deg(fov_h)))
            print("ver. FoV: {}".format(np.rad2deg(fov_v)))

            R = R_cam.T
            t = -R.dot(cam_pos)

            trans_2d = np.array([
                [1, 0, x0],
                [0, 1, y0],
                [0, 0, 1],
                ])
            scale_2d = np.array([
                [fx, 0, 0],
                [0, fy, 0],
                [0, 0, 1],
                ])
            shear_2d = np.array([
                [1, s/fx, 0],
                [0, 1, 0],
                [0, 0, 1],
                ])
            trans_3d = np.hstack([np.eye(3), t[:, None]])
            rot_3d = np.vstack([np.hstack([R, np.zeros((3, 1))]),
                                np.array([0, 0, 0, 1])[None, :]])

            K = trans_2d.dot(scale_2d).dot(shear_2d)  # Intrinsic matrix
            E = trans_3d.dot(rot_3d)                  # Extrinsic matrix

            P = K.dot(E)
            K_old, E_old = K, E
            K, E = recover_intrinsic_extrinsic(P)
            print("K, E reconstr diff. MSE: {}, {}".format(np.mean((K-K_old)**2), np.mean((E-E_old)**2)))

            # Points to show in homogenous coordinates.
            p = np.array([
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, -1, 1, 1],
                [-1, -1, 1, 1],
                [0, -1, 1, 1],
                ], dtype=float).T

            # Project world points to image
            p_image = project(p, K, E)
            # And project them back to the world again to verify that backprojection works.
            p_bp = backproject(p_image, K, R, t, n=self.plane_normal, p0=self.plane_point)

            # Project grid of image coordinates onto world
            p_im_grid = np.reshape(np.meshgrid(range(0, self.img_w, 50), range(0, self.img_h, 50)), (2, -1))
            p_bp_grid = backproject(p_im_grid, K, R, t, n=self.plane_normal, p0=self.plane_point)

            cam_arrow_end = cam_pos + R_cam.dot(np.array([0, 0, 1]))

            if self.l_world is None:
                ax_world.scatter(p[0, :], p[1, :], p[2, :], c='blue')
                self.l_world = ax_world.scatter(p_bp[0, :], p_bp[1, :], p_bp[2, :], c='red')
                self.l_cam = ax_world.scatter(cam_pos[0], cam_pos[1], cam_pos[2],
                                         c='black')
                self.cam_arrow = Arrow3D([cam_pos[0], cam_arrow_end[0]],
                                    [cam_pos[1], cam_arrow_end[1]],
                                    [cam_pos[2], cam_arrow_end[2]],
                                    mutation_scale=2,
                                    lw=1, arrowstyle='-|>', color="black")
                ax_world.add_artist(self.cam_arrow)
                self.cam_frustum = Frustum(img_w, img_h)
                self.cam_frustum.add_to_axis(ax_world)
                self.l_image = ax_image.scatter(p_image[0, :], p_image[1, :], c='red')

                self.l_world_grid = ax_world.scatter(p_bp_grid[0, :], p_bp_grid[1, :], p_bp_grid[2, :], c='green')
                ax_image.scatter(p_im_grid[0, :], p_im_grid[1, :], c='green')

            # Update positions
            self.l_world._offsets3d = (p_bp[0, :], p_bp[1, :], p_bp[2, :])
            self.l_cam._offsets3d = ([cam_pos[0]], [cam_pos[1]], [cam_pos[2]])
            self.cam_arrow._verts3d = ([cam_pos[0], cam_arrow_end[0]],
                                       [cam_pos[1], cam_arrow_end[1]],
                                       [cam_pos[2], cam_arrow_end[2]])
            self.cam_frustum.update(cam_pos, K, R)
            self.l_image.set_offsets(p_image[:2, :].T)
            self.l_world_grid._offsets3d = (p_bp_grid[0, :], p_bp_grid[1, :], p_bp_grid[2, :])
            fig.canvas.draw_idle()

        s_x0.on_changed(update)
        s_y0.on_changed(update)
        s_f.on_changed(update)
        s_s.on_changed(update)
        s_tx.on_changed(update)
        s_ty.on_changed(update)
        s_tz.on_changed(update)
        s_rx.on_changed(update)
        s_ry.on_changed(update)
        s_rz.on_changed(update)

        update(None)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-matrix', '-c', type=str)
    parser.add_argument('--plane-normal', '-n', type=float, nargs=3, default=[0, 1, 0],
                        help='Normal of the plane to backproject points onto.')
    parser.add_argument('--plane-point', '-p', type=float, nargs=3, default=[0, 2, 0],
                        help='Point on the plane to backproject points onto.')
    parser.add_argument('--world_boundary', '-w', type=float, default=5,
                        help='Limits of the world coordinates.')
    args = parser.parse_args()

    vis = PinholeCameraVisualization(640, 480, args.camera_matrix, args.plane_normal,
                                     args.plane_point, args.world_boundary)


if __name__ == '__main__':
    main()
