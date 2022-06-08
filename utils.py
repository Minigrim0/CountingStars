import pyopencl as cl


def load_kernels(context):
    """Loads the kernels from the OpenCL source files."""
    with open("./src/grayscale.cl", "r") as gs_kernel:
        kernel = gs_kernel.read()
        gs_kern = cl.Program(context, kernel).build()

    with open("./src/star_counter.cl", "r") as cs_kernel:
        kernel = cs_kernel.read()
        count_kern = cl.Program(context, kernel).build()

    return gs_kern, count_kern


def pad_up(group_size: tuple, global_size: tuple):
    """Pads the second tuple up to the nearest (upper) multiple of the first tuple."""
    diffs = (
        global_size[0] % group_size[0],
        global_size[1] % group_size[1])
    return (
        global_size[0] + (diffs[0] != 0) * (group_size[0] - diffs[0]),
        global_size[1] + (diffs[1] != 0) * (group_size[1] - diffs[1]))
