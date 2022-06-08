__constant float4 coefs = {0.2989f, 0.5870f, 0.1140f, 0};

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void to_grayscale(
    __read_only image2d_t src,
    __write_only image2d_t dst,
    int img_width,
    int img_height
) {
    __const int2 glob_id = {get_global_id(0), get_global_id(1)};

    // If all the coordinates are smaller than the size of their dimension, continue
    if(!all(glob_id < (int2){img_width, img_height})) return;

    float4 rgba = read_imagef(src, smp, glob_id);
    float gray = dot(rgba, coefs);
    write_imagef(dst, glob_id, (float4){gray, gray, gray, 255});
}