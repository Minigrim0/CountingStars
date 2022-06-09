import sys
import json
from matplotlib import pyplot as plt


# Open results json file (given by argv) and read data
with open(sys.argv[1], 'r') as f:
    json_dump = json.load(f)

def average_total(json_dump):
    """create a bar graph that shows the mean of the total duration per work size"""
    for image_name, value in json_dump["results"].items():
        keys = []
        data = []
        for work_group_size, values in value["work_groups"].items():
            if work_group_size == "size":
                continue
            keys.append(work_group_size)
            data.append(sum(values['total']) / len(values['total']))

        plt.bar(keys, data, width=0.4, label=image_name)

    plt.legend()
    plt.ylabel("Duration (s)")
    plt.xlabel("Work group size (x, x)")
    plt.title(f"Average total duration per work group size - {json_dump['device_name']}")
    plt.show()


def average_local_mem(json_dump):
    """create a bar graph that shows the mean of the local memory kernel version duration per work size"""
    for image_name, value in json_dump["results"].items():
        keys = []
        data = []
        for work_group_size, values in value["work_groups"].items():
            if work_group_size == "size":
                continue
            keys.append(work_group_size)
            data.append(sum(values['local_kern']) / len(values['local_kern']))

        plt.bar(keys, data, width=0.4, label=image_name)

    plt.legend()
    plt.ylabel("Duration (ns)")
    plt.xlabel("Work group size (x, x)")
    plt.title(f"Average local memory kernel version duration per work group size - {json_dump['device_name']}")
    plt.show()


def average_global_mem(json_dump):
    """create a bar graph that shows the mean of the global memory kernel version duration per work size"""
    for image_name, value in json_dump["results"].items():
        keys = []
        data = []
        for work_group_size, values in value["work_groups"].items():
            if work_group_size == "size":
                continue
            keys.append(work_group_size)
            data.append(sum(values['global_kern']) / len(values['global_kern']))

        plt.bar(keys, data, width=0.4, label=image_name)

    plt.legend()
    plt.ylabel("Duration (ns)")
    plt.xlabel("Work group size (x, x)")
    plt.title(f"Average global memory kernel version duration per work group size - {json_dump['device_name']}")
    plt.show()


def average_grayscale(json_dump):
    """create a bar graph that shows the mean of the grayscale kernel duration per work size"""
    for image_name, value in json_dump["results"].items():
        keys = []
        data = []
        for work_group_size, values in value["work_groups"].items():
            if work_group_size == "size":
                continue
            keys.append(work_group_size)
            data.append(sum(values['grayscale_kern']) / len(values['grayscale_kern']))

        plt.bar(keys, data, width=0.4, label=image_name)

    plt.legend()
    plt.ylabel("Duration (ns)")
    plt.xlabel("Work group size (x, x)")
    plt.title(f"Average grayscale kernel duration per work group size - {json_dump['device_name']}")
    plt.show()


def plot_local_mem_against_global_mem(json_dump):
    """create a bar graph that shows the mean of the local memory kernel version duration per work size"""
    for image_name, value in json_dump["results"].items():
        keys = []
        data = [[], []]
        for work_group_size, values in value["work_groups"].items():
            if work_group_size == "size":
                continue
            keys.append(work_group_size)
            data[0].append(sum(values['local_kern']) / len(values['local_kern']))
            data[1].append(sum(values['global_kern']) / len(values['global_kern']))
        
        plt.plot(keys, data[0], color="blue", label=f"{image_name} local version")
        plt.plot(keys, data[1], color="red", label=f"{image_name} global version")

    plt.legend()
    plt.ylabel("Duration (ns)")
    plt.xlabel("Work group size (x, x)")
    plt.title(f"Comparison of local memory and global memory versions - {json_dump['device_name']}")
    plt.show()


average_total(json_dump)
average_local_mem(json_dump)
average_global_mem(json_dump)
average_grayscale(json_dump)
plot_local_mem_against_global_mem(json_dump)