import os
import glob
import json
import timeit
import numpy as np
import pyopencl as cl
from utils import load_kernels
from main import to_grayscale, count_stars_local_mem, count_stars_global_mem
from PIL import Image


def benchmark():
    # Get list of platforms, select the first.
    platforms = cl.get_platforms()
    platform = platforms[0]

    # Get list of devices, select the first.
    devices = platform.get_devices(device_type=cl.device_type.GPU)
    device = devices[0]

    # Create a context containing the platform and device.
    context = cl.Context(devices=[device],
        properties=[(cl.context_properties.PLATFORM, platform)])

    # Create a command queue to the device.
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    gs_kern, count_kern = load_kernels(context)

    images = sorted(glob.glob('./images/**/*.jpg', recursive=True))

    results = {}

    for image_path in images:
        src_image = Image.open(image_path).convert("RGBA")
        color_image = np.asarray(src_image)

        _path, image_name = os.path.split(image_path)
        image_name, _ext = os.path.splitext(image_name)

        results[image_name] = {
            "size": color_image.shape,
        }

        for work_size in ((4, 4), (8, 8), (16, 16), (32, 32)):
            results[image_name][work_size[0]] = {}

            results[image_name][work_size[0]]["local_kern"] = []
            results[image_name][work_size[0]]["global_kern"] = []
            results[image_name][work_size[0]]["grayscale_kern"] = []
            results[image_name][work_size[0]]["total"] = []

            for x in range(50):
                start = timeit.default_timer()
                gs_image, timing = to_grayscale(context, color_image, queue, gs_kern, work_size)
                results[image_name][work_size[0]]["grayscale_kern"].append(timing)
                results[image_name][work_size[0]]["local_kern"].append(count_stars_local_mem(context, gs_image, queue, count_kern, work_size)[1])
                results[image_name][work_size[0]]["global_kern"].append(count_stars_global_mem(context, gs_image, queue, count_kern, work_size)[1])
                end = timeit.default_timer()
                results[image_name][work_size[0]]["total"].append(end - start)

    with open("./results/results.json", "w+") as result_fd:
        json.dump(results, result_fd, indent=4)

if __name__ == "__main__":
    benchmark()