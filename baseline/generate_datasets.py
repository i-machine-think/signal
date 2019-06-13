import numpy as np
import cairo
from data.image import Image
from enums.image_property import ImageProperty

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


def generate_image(
        seed,
        horizontal_position,
        vertical_position,
        shape,
        color,
        size,
        property_to_change: ImageProperty):
    np.random.seed(seed)

    if property_to_change == ImageProperty.Shape:
        n = N_SHAPES - 1
    elif property_to_change == ImageProperty.Size:
        n = N_SIZES - 1
    elif property_to_change == ImageProperty.Color:
        n = N_COLORS - 1
    else:
        n = N_CELLS - 1

    new_horizontal_position = horizontal_position
    new_vertical_position = vertical_position
    new_shape = shape
    new_color = color
    new_size = size

    result_images = []
    value_to_change = 0

    for _ in range(n):
        data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
        surf = cairo.ImageSurface.create_for_data(
            data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
        ctx = cairo.Context(surf)
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        ctx.paint()
        
        shapes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
        colors = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
        sizes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]

        if property_to_change == ImageProperty.Shape:
            if value_to_change == shape:
                value_to_change += 1

            new_shape = value_to_change
        elif property_to_change == ImageProperty.Size:
            if value_to_change == size:
                value_to_change += 1

            new_size = value_to_change
        elif property_to_change == ImageProperty.Color:
            if value_to_change == color:
                value_to_change += 1

            new_color = value_to_change
        elif property_to_change == ImageProperty.HorizontalPosition:
            if value_to_change == horizontal_position:
                value_to_change += 1

            new_horizontal_position = value_to_change
        elif property_to_change == ImageProperty.VerticalPosition:
            if value_to_change == vertical_position:
                value_to_change += 1

            new_vertical_position = value_to_change

        # Random location
        shapes[new_horizontal_position][new_vertical_position] = new_shape
        colors[new_horizontal_position][new_vertical_position] = new_color
        sizes[new_horizontal_position][new_vertical_position] = new_size

        draw(shapes[new_horizontal_position][new_vertical_position],
             colors[new_horizontal_position][new_vertical_position],
             sizes[new_horizontal_position][new_vertical_position],
             new_vertical_position,
             new_horizontal_position,
             ctx)
        
        value_to_change += 1

        metadata = {"shapes": shapes, "colors": colors, "sizes": sizes}
        new_image = Image(shapes, colors, sizes, data, metadata)
        result_images.append(new_image)

    return result_images

image_properties = list(map(int, ImageProperty))
seed = 42

all_images = {}

for shape in range(N_SHAPES):
    for size in range(N_SIZES):
        for color in range(N_COLORS):
            for horizontal_position in range(N_CELLS):
                for vertical_position in range(N_CELLS):
                    for image_property in image_properties:
                        current_images = generate_image(seed, horizontal_position, vertical_position, shape, color, size, image_property)
                        all_images[f'{horizontal_position}{vertical_position}{shape}{color}{size}{image_property}'] = current_images

print(len(all_images.keys()))
print(all_images.keys())