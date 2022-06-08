import pyopencl as cl
import numpy as np
from PIL import Image
import glob
from utils import load_kernels, pad_up


def to_grayscale(context, color_image, queue, gs_kern, local_work_size):
    color_image_buffer = cl.image_from_array(context, color_image, 4)
    image_size = (color_image.shape[0], color_image.shape[1])

    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    gs_image_buffer = cl.Image(context, cl.mem_flags.WRITE_ONLY, fmt, shape=image_size)
    gs_image = np.empty_like(color_image)

    global_work_size = pad_up(local_work_size, image_size)

    gs_ev = gs_kern.to_grayscale(queue, global_work_size, local_work_size, color_image_buffer, gs_image_buffer, np.int32(image_size[0]), np.int32(image_size[1]))
    gs_ev.wait()

    origin = ( 0, 0, 0 )
    region = (image_size[0], image_size[1], 1)

    cl._enqueue_read_image(queue, gs_image_buffer,
						origin, region, gs_image).wait()

    # Axes have a tendency to invert themselves
    return gs_image.reshape(image_size[1], image_size[0], 4), gs_ev.profile.end - gs_ev.profile.start


def count_stars_local_mem(context, source_image, queue, count_kern, local_work_size):
    """Count stars in an image by copying current WG region into local memory

    Args:
        context: The openCL context
        source_image (np.ndarray): the source image
        queue: The OpenCL Queue object
        count_kern: The kernel program to run from

    Returns:
        int: The amount of nanoseconds the kernel took to run
    """
    source_image_buffer = cl.image_from_array(context, source_image, 4)
    image_size = (source_image.shape[0], source_image.shape[1])

    counter = np.zeros(1).astype(np.int32)  # Counter int32
    counter_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, counter.nbytes, hostbuf=counter)

    global_work_size = pad_up(local_work_size, image_size)

    # Create local memory buffer for the kernel (Add two on each axis to account for the border)
    local_memory_buffer = cl.LocalMemory((local_work_size[0] + 2) * (local_work_size[1] + 2) * np.int32().nbytes)

    cs_ev = count_kern.count_stars_local_mem_copy(queue, global_work_size, local_work_size, source_image_buffer, local_memory_buffer, counter_buffer, np.int32(image_size[0]), np.int32(image_size[1]))
    cs_ev.wait()
    cl.enqueue_copy(queue, counter, counter_buffer)
    return counter, cs_ev.profile.end - cs_ev.profile.start


def count_stars_global_mem(context, source_image, queue, count_kern, local_work_size):
    source_image_buffer = cl.image_from_array(context, source_image, 4)
    image_size = (source_image.shape[0], source_image.shape[1])

    counter = np.zeros(1).astype(np.int32)  # Counter int32
    counter_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, counter.nbytes, hostbuf=counter)

    global_work_size = pad_up(local_work_size, image_size)

    cs_ev = count_kern.count_stars_global_mem(queue, global_work_size, local_work_size, source_image_buffer, counter_buffer, np.int32(image_size[0]), np.int32(image_size[1]))
    cs_ev.wait()
    cl.enqueue_copy(queue, counter, counter_buffer)
    return counter, cs_ev.profile.end - cs_ev.profile.start


def main(device_type):

    # Get list of platforms, select the first.
    platforms = cl.get_platforms()
    platform = platforms[0]

    # Get list of devices, select the first.
    devices = platform.get_devices(device_type=device_type)
    device = devices[0]

    # Create a context containing the platform and device.
    context = cl.Context(devices=[device],
        properties=[(cl.context_properties.PLATFORM, platform)])

    # Create a command queue to the device.
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    gs_kern, count_kern = load_kernels(context)

    images = sorted(glob.glob('./images/**/*.jpg', recursive=True))

    work_size = (32, 32)

    for image_path in images:
        src_image = Image.open(image_path).convert("RGBA")
        color_image = np.asarray(src_image)
        print(f"{' ' * 5}--=={{}}==--{' ' * 5}\nWorking on : {image_path} ({color_image.shape[0]}, {color_image.shape[1]})")

        gs_image, time = to_grayscale(context, color_image, queue, gs_kern, work_size)
        print(f"Grayscale kernel ex time: {time} ns")

        counter, time = count_stars_local_mem(context, gs_image, queue, count_kern, work_size)
        print(f"LOCALMEM Star counter: {str(time).zfill(12)} ns ({str(counter[0]).zfill(6)} stars)")

        counter, time = count_stars_global_mem(context, gs_image, queue, count_kern, work_size)
        print(f"GLOBMEM  Star counter: {str(time).zfill(12)} ns ({str(counter[0]).zfill(6)} stars)")

if __name__ == "__main__":
    main(device_type=cl.device_type.GPU)
