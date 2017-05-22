#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def rotx(t):
    """Creates a matrix that rotates t radians around the x axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Creates a matrix that rotates t radians around the y axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Creates a matrix that rotates t radians around the z axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def rot_matrix(rx, ry, rz):
    return rotz(rz).dot(roty(ry)).dot(rotx(rx))


def yaw_pitch_roll_from_rot_matrix(R):
    """Recovers yaw, pitch roll from a rotation matrix.

    Assumes rotation to have been
    applied in the order R = R_z * R_y * R_x.
    """
    yaw = np.arctan(R[1, 0] / R[0, 0])
    pitch = np.arctan(-R[2, 0] / np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = np.arctan(R[2, 1] / R[2, 2])
    return yaw, pitch, roll
