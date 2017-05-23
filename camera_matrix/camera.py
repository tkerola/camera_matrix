#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def rq(M):
    # RQ decomposition on matrix M.
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = np.fliplr(np.flipud(R.T))
    Q = np.flipud(Q.T)
    return R, Q


def recover_intrinsic_extrinsic(P):
    """Recovers the intrinsic and extrinsic matrices.

    Based on: http://ksimek.github.io/2012/08/14/decompose/

    Keyword args:
    - P: Camera matrix. shape = (3, 4)

    Returns:
    - K: Intrinsic matrix. shape = (3, 3)
    - E: Extrinsic matrix. shape = (3, 4)
    """
    M = P[:, :3]
    c = -np.linalg.inv(M).dot(P[:, 3])  # camera position
    K, R = rq(M)

    # Make diagonal entries positive.
    T = np.diag(np.sign(np.diag(K)))
    K = K.dot(T)
    R = T.dot(R)

    # # Does the x axis point in the positive direction?
    # if R[0, 0] < 0:
    #     print("flip x")
    #     # No, flip it.
    #     K[:, 0] *= -1
    #     R[0, :] *= -1
    # # Does the y axis point in the positive direction?
    # if R[1, 1] < 0:
    #     print("flip y")
    #     # No, flip it.
    #     K[:, 1] *= -1
    #     R[1, :] *= -1
    # # Does the z axis point in the positive direction?
    # if R[2, 2] < 0:
    #     print("flip z")
    #     # No, flip it.
    #     K[:, 2] *= -1
    #     R[2, :] *= -1

    if np.linalg.det(R) < 0:
        warnings.warn("The determinant of R was flipped during recovery.", RuntimeWarning)
        R *= -1  # Make sure that the determinant of R is positive

    E = np.hstack([R, -R.dot(c)[:, None]])
    return K, E


def project(p, K, E):
    """Projects point(s) p from world coordinates into the image plane.

    Keyword args:
    - p: Point(s) to project. shape = (3, n)
    - K: Intrinsic matrix of the camera. shape = (3, 3)
    - E: Extrinsic matrix of the camera. shape = (3, 4)

    Returns:
    - p_image: Point(s) projected to the image plane. shape = (2, n)
    """
    # print("Projecting {} onto image".format(p))
    if len(p.shape) != 2:
        p = p[:, None]
    if p.shape[0] != 4:
        p = np.vstack([p, np.ones(p.shape[1])])
    u = K.dot(E).dot(p)
    # print("K")
    # print(K)
    # print("E")
    # print(E)
    # print("u before divide:\n{}".format(u))
    u = u[:2, :] / u[2, :]  # If u[2, :] is negative, then the points are not visible.
    # print("Result is u =\n{}".format(u))
    return u


def backproject(u, K, R, t, n, p0, keep_only_pos_z=False):
    """Backprojects the image coordinates u onto a plane.

    Keyword args:
    - u: Image coordinates to project. shape = (2, n)
    - K: Intrinsic matrix of the camera. shape = (3, 3)
    - R: World rotation matrix from the camera. shape = (3, 3)
    - t: World translation vector from the camera. shape = (3,)
    - n: The normal vector of the plane. shape = (3,)
    - p0: A known point on the plane. shape = (3,)

    Returns:
    - p: Image coordinate backprojected onto the world. shape = (3, n)
    """
    # print("Backprojecting {} onto a plane".format(u))
    if len(u.shape) != 2:
        u = u[:, None]
    if u.shape[0] != 3:
        u = np.vstack([u, np.ones(u.shape[1])])
    if len(t.shape) != 2:
        t = t[:, None]
    n = np.asarray(n)

    Kinv = np.linalg.inv(K)
    c = -R.T.dot(t)
    d = n.dot(p0)

    RKinvp = R.T.dot(Kinv).dot(u)
    p = (d - n.dot(c)) / n.dot(RKinvp)[None, :] * RKinvp + c
    if keep_only_pos_z:
        p = R.dot(p) + t
        p = p[:, p[2, :] > 0]
        p = R.T.dot(p - t)
    # print("Result is p ({}) =\n{}".format(p.shape, p))
    return p


