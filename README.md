# Perlin Noise Wang Tile Generator
This script combines the concepts of [Perlin Noise](https://en.wikipedia.org/wiki/Perlin_noise) and [Wang Tiles](https://en.wikipedia.org/wiki/Wang_tile) to generate a set of non-periodically tileable noise images.

Based on the work of [David Maung](https://etd.ohiolink.edu/apexprod/rws_etd/send_file/send?accession=osu1461077485&disposition=inline), [Michael Cohen, Jonathan Shade, Stefan Hiller, and Oliver Deussen](https://www.researchgate.net/publication/2864579_Wang_Tiles_for_Image_and_Texture_Generation).

## Examples
![Tileable Perlin noise](/images/example1.png)
![With colorized edges](/images/example2.gif)

## Usage
```
usage: tile_generator.py [-h] [--output OUTPUT] [--output-name OUTPUT_NAME] [--output-format OUTPUT_FORMAT] [--seed SEED] --tile-size TILE_SIZE [--frequency FREQUENCY] [--period PERIOD]
                         [--amplitude AMPLITUDE] [--octaves OCTAVES] [--x-colors X_COLORS] [--y-colors Y_COLORS] [--n-choices N_CHOICES] [--colorize-edges]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to output directory
  --output-name OUTPUT_NAME
                        Name to prepend to all output files. Default = tile
  --output-format OUTPUT_FORMAT
                        Output file type. Default = png
  --seed SEED           Seed for random number generation. Default = random seed

Perlin noise settings:
  --tile-size TILE_SIZE
                        Size of tiles (length of one side in pixels, must be a power of 2)
  --frequency FREQUENCY
                        Frequency of Perlin lattice points in pixels, overrides --period. Default = None
  --period PERIOD       Number of pixels between Perlin lattice points; inverse of frequency. Default = 8
  --amplitude AMPLITUDE
                        Delta between min and max value. Default = 1.0
  --octaves OCTAVES     Number of noise waves to combine; a higher value results in more detail. Default = 1

Wang tile settings:
  Total number of tiles will be x_colors * y_colors * n_choices.

  --x-colors X_COLORS   Number of colors to use when tiling left-right. Default = 2
  --y-colors Y_COLORS   Number of colors to use when tiling up-down. Default = 2
  --n-choices N_CHOICES
                        Number of alternative tile options for each unique pair of tile up-left colors. Default = 2
  --colorize-edges      Colorize edges in output for easier visual matching. Desaturate to restore original images.

```

## Dependencies
* [more_itertools](https://more-itertools.readthedocs.io/en/stable/)
* [numpy](https://numpy.org/doc/stable/index.html)
* [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/index.html)
* [vectormath](https://pypi.org/project/vectormath/)

