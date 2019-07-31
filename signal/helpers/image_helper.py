import numpy as np
import cairo
from data.image import Image

N_CELLS = 3

WIDTH = 30
HEIGHT = 30

CELL_WIDTH = WIDTH / N_CELLS
CELL_HEIGHT = HEIGHT / N_CELLS
N_CHANNELS = 3

BIG_RADIUS = CELL_WIDTH * 0.75 / 2
SMALL_RADIUS = CELL_WIDTH * 0.5 / 2

SHAPE_CIRCLE = 0
SHAPE_SQUARE = 1
SHAPE_TRIANGLE = 2
N_SHAPES = SHAPE_TRIANGLE + 1

SIZE_SMALL = 0
SIZE_BIG = 1
N_SIZES = SIZE_BIG + 1

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
N_COLORS = COLOR_BLUE + 1


def draw(shape, color, size, left, top, ctx):
    center_x = (left + 0.5) * CELL_WIDTH
    center_y = (top + 0.5) * CELL_HEIGHT

    radius = SMALL_RADIUS if size == SIZE_SMALL else BIG_RADIUS
    radius *= 0.9 + np.random.random() * 0.2

    if color == COLOR_RED:
        rgb = np.asarray([1.0, 0.0, 0.0])
    elif color == COLOR_GREEN:
        rgb = np.asarray([0.0, 1.0, 0.0])
    else:
        rgb = np.asarray([0.0, 0.0, 1.0])
    rgb += np.random.random(size=(3,)) * 0.4 - 0.2
    rgb = np.clip(rgb, 0.0, 1.0)

    if shape == SHAPE_CIRCLE:
        ctx.arc(center_x, center_y, radius, 0, 2 * np.pi)
    elif shape == SHAPE_SQUARE:
        ctx.new_path()
        ctx.move_to(center_x - radius, center_y - radius)
        ctx.line_to(center_x + radius, center_y - radius)
        ctx.line_to(center_x + radius, center_y + radius)
        ctx.line_to(center_x - radius, center_y + radius)
    else:
        ctx.new_path()
        ctx.move_to(center_x - radius, center_y + radius)
        ctx.line_to(center_x, center_y - radius)
        ctx.line_to(center_x + radius, center_y + radius)
    ctx.set_source_rgb(*rgb)
    ctx.fill()


def get_image(seed, shape=-1, color=-1, n=1, nOtherShapes=0, shouldOthersBeSame=False):
    np.random.seed(seed)

    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.paint()

    shapes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    colors = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    sizes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]

    shape = shape if shape >= 0 else np.random.randint(N_SHAPES)
    color = color if color >= 0 else np.random.randint(N_COLORS)

    for _ in range(n):
        # Random location
        r = np.random.randint(N_CELLS)
        c = np.random.randint(N_CELLS)

        shapes[r][c] = shape
        colors[r][c] = color
        sizes[r][c] = np.random.randint(N_SIZES)

        draw(shapes[r][c], colors[r][c], sizes[r][c], c, r, ctx)

    metadata = {"shapes": shapes, "colors": colors, "sizes": sizes}

    return Image(shapes, colors, sizes, data, metadata)


if __name__ == "__main__":
    i = get_image(42)
    print(i)
