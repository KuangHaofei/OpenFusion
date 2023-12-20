import numpy as np

from openfusion.utils import rand_cmap, save_cmap_legend_bar_separated

np.random.seed(42)

query = [
    "floor",
    "wall",
    "door",
    "window",
    "sofa",
    "bed",
    "chair",
    "light",
    "table",
    "cabinet",
    "refrigerator",
    "air_conditioner",
    "kitchen_table",
    "tv",
    "ball",
    "others"
]

cmap = rand_cmap(len(query), type="bright", first_color_black=False)
save_cmap_legend_bar_separated(cmap, query)
