class Image:
    def __init__(self, shapes, colors, sizes, data, metadata):
        self.shapes = shapes
        self.colors = colors
        self.sizes = sizes
        self.data = data
        self.metadata = metadata

    def __str__(self):
        return str(self.metadata)