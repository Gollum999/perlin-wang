import math
import random
from typing import Self

import numpy as np
from vectormath import Vector2

from typedefs import ColorEdges, ColorToEdgeDict, EdgeColors, Rect


def _make_gradients(count: int):
    """ Build a list of gradient vectors for Perlin noise. """
    return [Vector2(math.cos(arc_idx * 2.0 * math.pi / count),
                    math.sin(arc_idx * 2.0 * math.pi / count))
            for arc_idx in range(count)]


class PerlinLattice:
    """
    This represents the "lattice" of gradients that ultimately gives Perlin noise its random appearance.
    Edge and corner gradients are overridden to enable seamless tiling.
    """

    GRADIENT_COUNT = 256
    GRADIENTS = _make_gradients(GRADIENT_COUNT)

    def __init__(self, color_to_edge: ColorToEdgeDict, color_indices: EdgeColors):
        self.center_idxs = self._make_center_permutations()
        self.edge_idxs = self._make_edge_permutations(color_to_edge, color_indices)
        # all corners must share the same gradient index since the corner points are shared between edges
        self.corner_idx = color_to_edge[0][0]

    def _make_center_permutations(self) -> list[int]:
        """ Build the primary "permutation set" used to choose pseudo-random gradients at each lattice point. """
        center_permutations = list(range(self.GRADIENT_COUNT))
        random.shuffle(center_permutations)
        center_permutations += center_permutations
        return center_permutations

    def _make_edge_permutations(self, color_to_edge: ColorToEdgeDict, color_indices: EdgeColors) -> ColorEdges:
        """ Build the "permutation sets" that override all edge points of the lattice. """
        return ColorEdges(
            top=color_to_edge[color_indices.top],
            right=color_to_edge[color_indices.right],
            bottom=color_to_edge[color_indices.bottom],
            left=color_to_edge[color_indices.left],
        )

    def get_all_corners(self, lattice_size: int) -> list[Vector2]:
        """ Return the coordinates that represent the corners of this lattice. """
        return [
            Vector2(0, 0),
            Vector2(0, lattice_size - 1),
            Vector2(lattice_size - 1, 0),
            Vector2(lattice_size - 1, lattice_size - 1),
        ]

    def get_gradient_vector(self, lattice_point: Vector2, lattice_size: int) -> Vector2:
        """ Get the gradient at the specified lattice point. """
        edges = Rect(top=0, left=0, right=lattice_size - 1, bottom=lattice_size - 1)
        match tuple(lattice_point.astype(int)):
            # corners
            case (edges.left, edges.top) | (edges.right, edges.top) | (edges.left, edges.bottom) | (edges.right, edges.bottom):
                lattice_point_hash = self.corner_idx

            # edges
            case (edges.left, y):
                lattice_point_hash = self.edge_idxs.left[y]
            case (edges.right, y):
                lattice_point_hash = self.edge_idxs.right[y]
            case (x, edges.top):
                lattice_point_hash = self.edge_idxs.top[x]
            case (x, edges.bottom):
                lattice_point_hash = self.edge_idxs.bottom[x]

            # center
            case (x, y):
                lattice_point_hash = self.center_idxs[self.center_idxs[x % self.GRADIENT_COUNT] + y % self.GRADIENT_COUNT]

        return self.GRADIENTS[lattice_point_hash]

    def gradient(self, point: Vector2, lattice_point: Vector2, lattice_size: int) -> float:
        """ Calculate the partial noise value at point based on the gradient vector of the specified lattice point. """
        assert list(lattice_point) == list(np.floor(lattice_point))  # corner always falls on int coords
        delta_from_corner = point - lattice_point
        corner_gradient = self.get_gradient_vector(lattice_point, lattice_size)
        return (self.smooth_ramp(abs(delta_from_corner.x))
              * self.smooth_ramp(abs(delta_from_corner.y))
              * delta_from_corner.dot(corner_gradient))

    @staticmethod
    def smooth_ramp(t: float) -> float:
        """ Quintic polynomial smoothing. Called 'fade' in the reference implementation. """
        assert 0.0 <= t <= 1.0
        return 1 - (6 * t**5) + (15 * t**4) - (10 * t**3)
