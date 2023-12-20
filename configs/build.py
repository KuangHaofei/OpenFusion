import os.path as osp
from openfusion.datasets import HabitatSim, Robot

BASE_PATH = osp.dirname(osp.dirname(osp.abspath(__file__)))

PARAMS = {
    "habitat": {
        "dataset": HabitatSim,
        "path": "{}/sample/habitat/{}",
        "depth_scale": 1000.0,
        "depth_max": 5.0,
        "voxel_size": 8.0 / 512,
        "block_resolution": 8,
        "block_count": 200000,
        "img_size": (1080, 720),
        "input_size": (1080, 720)
    },
    "house": {
        "dataset": Robot,
        "path": "{}/sample/house/{}",
        "depth_scale": 1000.0,
        "depth_max": 3.0,
        "voxel_size": 6.0 / 512,
        "block_resolution": 8,
        "block_count": 500000,
        "img_size": (640, 480),
        "input_size": (640, 480),
        "objects": [
            "ceiling",
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
    },
    "office": {
        "dataset": Robot,
        "path": "{}/sample/office/{}",
        "depth_scale": 1000.0,
        "depth_max": 3.0,
        "voxel_size": 6.0 / 512,
        "block_resolution": 8,
        "block_count": 100000,
        "img_size": (640, 480),
        "input_size": (640, 480),
        "objects": [
            'ceiling',
            'floor',
            'wall',
            'sink',
            'door',
            'oven',
            'garbage can',
            'whiteboard',
            'table',
            'desk',
            'sofa',
            'chair',
            'bookshelf',
            'cabinet',
            'extinguisher',
            'people',
            'others'
        ]
    },
    "hospital": {
        "dataset": Robot,
        "path": "{}/sample/hospital/{}",
        "depth_scale": 1000.0,
        "depth_max": 5.0,
        "voxel_size": 12.0 / 512,
        "block_resolution": 8,
        "block_count": 500000,
        "img_size": (640, 480),
        "input_size": (640, 480),
        "objects": [
            "ceiling",
            "floor",
            "walls",
            "nurses station",
            "door",
            "chair",
            "trolley bed",
            "table",
            "sofa",
            "medical machine",
            "tv",
            "kitchen cabinet",
            "refrigerator",
            "toilet",
            "sink",
            "trash",
            "warehouse clusters",
            "others"
        ]
    },
}


def get_config(dataset, scene):
    params = PARAMS[dataset].copy()
    params["path"] = params["path"].format(BASE_PATH, scene)
    return params
