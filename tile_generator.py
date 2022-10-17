#!/usr/bin/env python3
import argparse
import colorsys
import itertools
import logging
import random
import sys
from pathlib import Path
from typing import Iterable, NamedTuple, TypeVar

import more_itertools
import numpy as np
from PIL import Image, ImageDraw
from vectormath import Vector2

from perlin_lattice import PerlinLattice
from typedefs import ColorEdge, ColorToEdgeDict, EdgeColors


logger = logging.getLogger(__name__)


class PerlinArgs(NamedTuple):
    """ Bundle of args for controlling Perlin noise. """
    tile_size: int
    frequency: float
    amplitude: float
    octaves: int

    def lattice_size(self, octave: int) -> int:
        """ Return the edge length of the Perlin lattice that is required to calculate noise at the specified octave. """
        size = int(self.tile_size * self.frequency * 2**octave) + 1
        assert is_power_of_two(size - 1)
        return size


class WangArgs(NamedTuple):
    """ Bundle of args for controlling Wang tiles. """
    x_colors: int
    y_colors: int
    n_choices: int
    colorize_edges: bool


def is_power_of_two(n: int) -> bool:
    """ Return True if `n` is a power of 2. """
    return (n != 0) and (n & (n-1) == 0)


def parse_args() -> argparse.Namespace:
    """ Parse command-line arguments and validate values. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Path to output directory')
    parser.add_argument('--output-name', type=str, default='tile', help='Name to prepend to all output files. Default = %(default)s')
    parser.add_argument('--output-format', type=str, default='png', help='Output file type. Default = %(default)s')
    parser.add_argument('--seed', type=int, help='Seed for random number generation. Default = random seed')

    perlin_group = parser.add_argument_group('Perlin noise settings')
    perlin_group.add_argument('--tile-size', type=int, required=True,
                              help='Size of tiles (length of one side in pixels, must be a power of 2)')
    perlin_group.add_argument('--frequency', type=float,
                              help='Frequency of Perlin lattice points in pixels, overrides --period. Default = %(default)s')
    perlin_group.add_argument('--period', type=int, default=8,
                              help='Number of pixels between Perlin lattice points; inverse of frequency. Default = %(default)s')
    perlin_group.add_argument('--amplitude', type=float, default=1.0,
                              help='Delta between min and max value. Default = %(default)s')
    perlin_group.add_argument('--octaves', type=int, default=1,
                              help='Number of noise waves to combine; a higher value results in more detail. Default = %(default)s')

    wang_group = parser.add_argument_group('Wang tile settings',
                                           description='Total number of tiles will be x_colors * y_colors * n_choices.')
    wang_group.add_argument('--x-colors', type=int, default=2,
                            help='Number of colors to use when tiling left-right. Default = %(default)s')
    wang_group.add_argument('--y-colors', type=int, default=2,
                            help='Number of colors to use when tiling up-down. Default = %(default)s')
    wang_group.add_argument('--n-choices', type=int, default=2,
                            help='Number of alternative tile options for each unique pair of tile up-left colors. Default = %(default)s')
    wang_group.add_argument('--colorize-edges', action='store_true',
                            help='Colorize edges in output for easier visual matching. Desaturate to restore original images.')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randrange(sys.maxsize)

    if args.frequency is None:
        if args.period is None:
            parser.error('Either --frequency or --period is required')
        else:
            args.frequency = 1.0 / args.period
    else:
        args.period = None  # override the default so we can detect which arg was specified

    if args.tile_size < 1:
        parser.error('--tile-size must be >= 1')
    if not is_power_of_two(args.tile_size):
        parser.error('--tile-size must be a power of 2')
    if args.period is not None:
        if args.period < 1:
            parser.error('--period must be >= 1')
        if not is_power_of_two(args.period):
            parser.error('--period must be a power of 2')
    else:
        if not 0.0 <= args.frequency <= 1.0:
            parser.error('--frequency must be between 0.0 and 1.0')
        if not (1.0 / args.frequency).is_integer() or not is_power_of_two(int(1.0 / args.frequency)):
            parser.error('--frequency must be the inverse of a power of 2')
    if not 0.0 <= args.amplitude <= 1.0:
        parser.error('--amplitude must be between 0.0 and 1.0')
    if args.octaves < 1:
        parser.error('--octaves must be >= 1')

    if args.x_colors < 2:
        parser.error('--x-colors must be >= 2')
    if args.y_colors < 2:
        parser.error('--y-colors must be >= 2')
    if args.n_choices < 2:
        parser.error('--n-choices must be >= 2')

    return args


def main():
    args = parse_args()
    logging.basicConfig(
        # level=logging.INFO,
        level=logging.DEBUG,
        format='%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s',
    )

    logger.info(f'Seed: {args.seed}')
    random.seed(args.seed)

    perlin_args = PerlinArgs(args.tile_size, args.frequency, args.amplitude, args.octaves)
    wang_args = WangArgs(args.x_colors, args.y_colors, args.n_choices, args.colorize_edges)

    logger.info(f'Creating output directory if necessary: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)

    write_tiled_perlin_images(args.output, args.output_name, args.output_format, perlin_args, wang_args)


def write_tiled_perlin_images(output_dir: Path, output_name: str, output_format: str, perlin: PerlinArgs, wang: WangArgs) -> Image:
    """ Generate all Perlin noise Wang tile images and write them to the specified directory. """
    color_count = wang.x_colors + wang.y_colors
    tile_count = wang.x_colors * wang.y_colors * wang.n_choices
    max_lattice_size = perlin.lattice_size(perlin.octaves - 1)

    # edge vectors of the same "color" need to stay constant across all tiles
    color_to_edge = build_color_to_edge_mapping(color_count, max_lattice_size)

    logger.info(f'Generating {tile_count} images of size {perlin.tile_size}x{perlin.tile_size}')
    combined_size = (wang.x_colors * wang.n_choices * perlin.tile_size, wang.y_colors * perlin.tile_size)
    combined_img = Image.new(mode='RGBA' if wang.colorize_edges else 'L', size=combined_size)
    all_color_combos = list(get_all_wang_colors(wang))
    logger.debug(f'All tile colors: {all_color_combos}')
    assert len(all_color_combos) == tile_count
    assert sorted(set(all_color_combos)) == sorted(all_color_combos)  # check for duplicates

    for idx, color_indices in enumerate(all_color_combos):
        perlin_lattice = PerlinLattice(color_to_edge, color_indices)

        output_file = output_dir / f'{output_name}_{"_".join(str(i) for i in color_indices)}.{output_format.lower()}'
        logger.info(f'Generating {output_file}')
        img = generate_perlin_image(perlin, wang, perlin_lattice)

        if wang.colorize_edges:
            img = colorize(img, color_count, color_indices)

        logger.info(f'Saving {output_file}')
        img.save(output_file)

        x_idx = idx % (wang.x_colors * wang.n_choices)
        y_idx = idx // (wang.x_colors * wang.n_choices)
        combined_img.paste(img, box=(x_idx * perlin.tile_size, y_idx * perlin.tile_size))

    combined_output_file = output_dir / f'{output_name}_combined.{output_format.lower()}'
    logger.info(f'Saving {combined_output_file}')
    combined_img.save(combined_output_file)

    logger.info('Done. Color indexes in filenames are in NESW (top-right-bottom-left) order.')


def build_color_to_edge_mapping(n_colors: int, max_lattice_size: int) -> ColorToEdgeDict:
    """ Build a mapping of color index to edge permutations. """

    def make_color_edge() -> ColorEdge:
        """ Build a single list of edge permutations. """
        assert max_lattice_size <= PerlinLattice.GRADIENT_COUNT, max_lattice_size
        return random.sample(range(0, PerlinLattice.GRADIENT_COUNT), max_lattice_size)

    return {color_idx: make_color_edge() for color_idx in range(n_colors)}


def get_all_wang_colors(wang: WangArgs) -> Iterable[EdgeColors]:
    """ Generate a (non-minimal) list of color combinations to create a non-periodic Wang tiling. """
    # No idea if this is a good algorithm, I just made something up
    tile_count = wang.x_colors * wang.y_colors * wang.n_choices
    color_pairs = list(itertools.product(range(wang.x_colors), range(wang.y_colors)))
    nw_colors = color_pairs * wang.n_choices
    rotation_amount = len(nw_colors) // 2 + 1
    se_colors = rotated(list(itertools.chain.from_iterable(itertools.repeat(colors, wang.n_choices) for colors in color_pairs)),
                        rotation_amount)

    # shift the up-down tile ids to be unique
    # mostly for clarity, but also allows us to use a single dict for colors rather than separate ones for NS and EW
    # technically it is just as correct to re-use colors between axes since tile rotation is disallowed when tiling
    nw_colors = [(left_color, up_color + wang.x_colors) for left_color, up_color in nw_colors]
    se_colors = [(right_color, down_color + wang.x_colors) for right_color, down_color in se_colors]
    logger.debug(f'North-west color list: {nw_colors}')
    logger.debug(f'South-east color list: {se_colors}')

    assert len(nw_colors) == len(se_colors)
    for (left_color, up_color), (right_color, down_color) in zip(nw_colors, se_colors):
        yield EdgeColors(top=up_color, right=right_color, bottom=down_color, left=left_color)


def rotated(l: list, n: int) -> list:
    """ Rotate the specified list `l` forward by `n` places. """
    return l[n:] + l[:n]


def generate_perlin_image(perlin: PerlinArgs, wang: WangArgs, lattice: PerlinLattice) -> Image:
    """ Build a single noise tile image. """
    img = Image.new(mode='RGBA' if wang.colorize_edges else 'L', size=(perlin.tile_size, perlin.tile_size))
    buf = list(img.getdata())
    for idx, px in enumerate(buf):
        point = Vector2(idx % perlin.tile_size, idx // perlin.tile_size)
        value = ((noise(point, perlin, wang, lattice) / 2.0) + 0.5) * 255
        buf[idx] = (int(value), int(value), int(value)) if wang.colorize_edges else value
    img.putdata(buf)
    return img


def noise(point: Vector2, perlin: PerlinArgs, wang: WangArgs, lattice: PerlinLattice) -> float:  # [-1.0, 1.0]
    """ Calculate Perlin noise at point with multiple octaves and overridden edge gradients. """
    # weighted sum of noise with increasing frequencies
    result = 0.0
    for octave in range(perlin.octaves):
        lattice_size = perlin.lattice_size(octave)

        # sanity check that all corners share the same gradient
        assert more_itertools.all_equal(lattice.get_gradient_vector(corner, lattice_size)
                                        for corner in lattice.get_all_corners(lattice_size))

        weight = perlin.amplitude / 2**octave
        result += weight * _noise(
            point=(point * perlin.frequency * 2**octave),
            lattice=lattice,
            lattice_size=lattice_size,
        )
    assert -1.0 <= result <= 1.0, result
    return result


def _noise(point: Vector2, lattice: PerlinLattice, lattice_size: int) -> float:  # [-1.0, 1.0]
    """ Calculate one octave of Perlin noise at point with overridden edge gradients. """
    top_left_corner = Vector2(np.floor(point))
    corners = [
        top_left_corner,
        top_left_corner + Vector2(1, 0),
        top_left_corner + Vector2(0, 1),
        top_left_corner + Vector2(1, 1),
    ]
    result = sum(lattice.gradient(point, corner, lattice_size) for corner in corners)

    assert -1.0 <= result <= 1.0, result
    return result


def colorize(img: Image, color_count: int, color_indices: EdgeColors) -> Image:
    """ Colorize the edges of the specified tile image for easier visual matching. """
    THICKNESS = 0.5  # full triangular "quadrant"
    TRAPEZOIDS = [
        [(0, 0), (1, 0), (1 - THICKNESS, THICKNESS), (THICKNESS, THICKNESS)],  # top
        [(1, 0), (1, 1), (1 - THICKNESS, 1 - THICKNESS), (1 - THICKNESS, THICKNESS)],  # right
        [(1, 1), (0, 1), (THICKNESS, 1 - THICKNESS), (1 - THICKNESS, 1 - THICKNESS)],  # bottom
        [(0, 1), (0, 0), (THICKNESS, THICKNESS), (THICKNESS, 1 - THICKNESS)],  # left
    ]

    color_map = make_color_map(color_count, saturation=0.5)

    color_img = Image.new('RGBA', size=img.size)
    draw = ImageDraw.Draw(color_img)
    for side_idx, trapezoid in enumerate(TRAPEZOIDS):
        color = color_map[color_indices[side_idx]]
        scaled_trapezoid = [tuple(Vector2(vertex) * Vector2(img.size)) for vertex in trapezoid]
        draw.polygon(scaled_trapezoid, fill=tuple(int(c * 255) for c in color))

    return blend_color(img, color_img)


def make_color_map(color_count: int, saturation: float) -> dict[int, tuple[int, int, int]]:
    """ Build a mapping of color index to a unique RGB triple. Distribute hues as evenly as possible to (ideally) avoid confusion. """
    hues = np.linspace(0.0, 1.0, num=color_count, endpoint=False)
    assert len(hues) == color_count
    return {color_idx: (*colorsys.hsv_to_rgb(hues[color_idx], saturation, 1.0), 1.0) for color_idx in range(color_count)}


def blend_color(background: Image, foreground: Image) -> Image:
    """ Blend the colors from `foreground` into the `background` image. """
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    bg_array = np.asarray(background).astype(float)  # dimensions: x, y, rgba
    fg_array = np.asarray(foreground).astype(float)

    bg_array_transposed = np.moveaxis(bg_array, source=2, destination=0)  # dimensions: rgba, x, y
    fg_array_transposed = np.moveaxis(fg_array, source=2, destination=0)

    bg_r, bg_g, bg_b, bg_a = bg_array_transposed
    fg_r, fg_g, fg_b, fg_a = fg_array_transposed

    bg_h, bg_s, bg_v = rgb_to_hsv(bg_r, bg_g, bg_b)
    fg_h, fg_s, fg_v = rgb_to_hsv(fg_r, fg_g, fg_b)

    # take hue/saturation from foreground, and take value from background
    out_r, out_g, out_b = hsv_to_rgb(fg_h, fg_s, bg_v)
    out_arr = np.dstack((out_r, out_g, out_b, bg_a))

    return Image.fromarray(out_arr.astype('uint8'), 'RGBA')


if __name__ == '__main__':
    main()
