# Counting Stars

This is the repository for the third project of the course *Multicore Programming* of the VUB.

## About

This project aims to count the amount of stars on an image, using OpenCL to parallelize the execution.

## Algorithm overview

* Transform the image to a grayscale image
* For each pixel :
    * Check if its luminescence is above 200
    * If so, check if at least 40% of its direct neighbours have a luminescence above 30
    * If this is the case, count this pixel as a star.

## Installation
* Clone the repositry
* Create a virtual environment
* `pip install -r requirements.txt`

### To run the main program
(Counting the stars of all images in the `images/` folder)

```
python3 main.py <WorkGroupSize>
```

### To run the benchmarks
(Couting the stars of all images in the `images/` folder with loads of configurations)

```
mkdir results
python3 benchmark.py
```

This will produce a `JSON` file with the results of the tests.