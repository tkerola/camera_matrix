#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from matplotlib.patches import FancyArrowPatch, Patch
from mpl_toolkits.mplot3d import proj3d

from camera import backproject

class Arrow3D(FancyArrowPatch):
    # http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Frustum(object):
    def __init__(self, img_w, img_h, length=4):
        self.img_w = img_w
        self.img_h = img_h
        self.length = length
        self.edge_inds = [(0, 1), (0, 2), (0, 3), (0, 4),
                          (1, 2), (2, 3), (3, 4), (4, 1)]
        self.arrows = []
        for i, j in self.edge_inds:
            self.arrows.append(Arrow3D(
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    mutation_scale=2,
                    lw=1, arrowstyle='-', color="black"
                    ))

    def update(self, pos, K, R=None):
        p = np.zeros((3, 4))
        p[:, 0] = [self.img_w, self.img_h, 1]
        p[:, 1] = [self.img_w, 0, 1]
        p[:, 2] = [0, 0, 1]
        p[:, 3] = [0, self.img_h, 1]
        p = backproject(p, K, np.eye(3), np.zeros(3), n=[0, 0, 1], p0=[0, 0, 1])
        p[:, :] *= self.length
        if R is not None:
            p = R.T.dot(p)
        p = np.hstack([pos[:, None], p])
        p[:, 1:] += p[:, 0][:, None]

        for a, (i, j) in zip(self.arrows, self.edge_inds):
            a._verts3d = ([p[0, i], p[0, j]],
                          [p[1, i], p[1, j]],
                          [p[2, i], p[2, j]])

    def add_to_axis(self, ax):
        for a in self.arrows:
            ax.add_artist(a)


def create_slider(pos, label, minval, maxval, initval):
    axcolor = 'lightgoldenrodyellow'
    ax = plt.axes(pos, facecolor=axcolor)
    s = Slider(ax, label, minval, maxval, valinit=initval)
    return s

