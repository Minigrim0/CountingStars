__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE
    | CLK_FILTER_NEAREST
    | CLK_ADDRESS_NONE;  // Ensures the values out of the image are 0
__constant uint2 thresholds = {200, 30};

/**
    Computes the position of the given 2D point in the local memory vector.
    @param[int2] local_coord The 2D point to compute the position for.
    @return The position of the given point in the local memory vector.
 */
int local_idx(int2 local_coord){
    // Add 1 to account for the border (which contains pixels from the previous chunk)
    return (local_coord.x + 1) + ((local_coord.y + 1) * (get_local_size(0) + 2));
}

/**
    Computes the global position of the given local position.
    @param[int2] local_coord The local position to compute the global position for.
    @return The global position of the given local position.
 */
int2 to_global_coord(int2 local_coord){
    return (int2){
        local_coord.x + (get_group_id(0) * get_local_size(0)),
        local_coord.y + (get_group_id(1) * get_local_size(1))};
}

/**
    Counts the stars in the given image.
    @param[image2d_t] src The image to count stars in.
    @return The number of stars in the given image.
 
    This implementation copies the image region to the local memory, then counts the stars.
 */
__kernel void count_stars_local_mem_copy(
    __read_only image2d_t src,
    __local uint* local_mem,  // Store image region in local memory
    __global int* counter,  // Actual counter
    int image_width,
    int image_height
){
    int2 local_coord = {get_local_id(0), get_local_id(1)};
    int2 global_coord = {get_global_id(0), get_global_id(1)};
    // Translate 2d local coordinate to 1d index
    int local_position = local_idx(local_coord);

    // Copy image region to local memory
    local_mem[local_position] = read_imageui(src, smp, global_coord).x;

    // If we're on the border of the x axis, we need to copy the previous chunk's pixels
    int2 ext_coord;
    if(local_coord.x == 0 && local_coord.y == 0){  // Diagonal upper-left corner
        ext_coord = (int2){local_coord.x - 1, local_coord.y - 1};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    }else if(local_coord.x == get_local_size(0)-1 && local_coord.y == 0){  // Diagonal upper-right corner
        ext_coord = (int2){local_coord.x + 1, local_coord.y - 1};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    }else if(local_coord.x == 0 && local_coord.y == get_local_size(1)-1){  // Diagonal lower-left corner
        ext_coord = (int2){local_coord.x - 1, local_coord.y + 1};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    }else if(local_coord.x == get_local_size(0) - 1 && local_coord.y == get_local_size(1) - 1){  // Diagonal lower-right corner
        ext_coord = (int2){local_coord.x + 1, local_coord.y + 1};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    }

    if(local_coord.x == 0){
        ext_coord = (int2){local_coord.x - 1, local_coord.y};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    } else if (local_coord.x == get_local_size(0) - 1){
        ext_coord = (int2){local_coord.x + 1, local_coord.y};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    }

    // Do the same for the y axis
    if(local_coord.y == 0){
        ext_coord = (int2){local_coord.x, local_coord.y - 1};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    } else if (local_coord.y == get_local_size(1) - 1){
        ext_coord = (int2){local_coord.x, local_coord.y + 1};
        local_mem[local_idx(ext_coord)] = read_imageui(src, smp, to_global_coord(ext_coord)).x;
    }

    // Wait for all threads to finish copying
    barrier(CLK_GLOBAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Count stars
    if(local_mem[local_position] > (uint)thresholds.x){  // Star must be a very bright spot (> 200)
        int neighbors = 0;
        int pixels = 0;
        int i;
        for(i=0;i<9;i++){
            if(i == 4) continue;
            neighbors += (int)(local_mem[local_idx(local_coord + (int2){i%3 - 1, i/3 - 1})] > (uint)thresholds.y);
            pixels += (int)(all(global_coord + (int2){i%3 - 1, i/3 - 1} >= (int2){0, 0}) && all(global_coord + (int2){i%3 - 1, i/3 - 1} < (int2){image_width, image_height}));
        }
        if((float)neighbors >= (float)pixels * 0.4)
            atomic_inc(counter);
    }
}

/**
    Counts the stars in the given image.
    @param[image2d_t] src The image to count stars in.
    @return The number of stars in the given image.
 
    This implementation only uses global memory to read the image.
 */
__kernel void count_stars_global_mem(
    __read_only image2d_t src,
    __global int* counter,  // Actual counter
    int image_width,
    int image_height
){
    // Count stars
    int2 gpos = {get_global_id(0), get_global_id(1)};
    uint pixel = read_imageui(src, smp, gpos).x;

    if(pixel > (uint)thresholds.x){
        int neighbors = 0;
        int pixels = 0;
        int i;
        for(i=0;i<9;i++){
            if(i == 4) continue;
            neighbors += (read_imageui(src, smp, gpos + (int2){i%3 - 1, i/3 - 1}).x > (uint)thresholds.y) ? 1 : 0;
            pixels += (all(gpos + (int2){i%3 - 1, i/3 - 1} >= (int2){0, 0}) && all(gpos + (int2){i%3 - 1, i/3 - 1} < (int2){image_width, image_height})) ? 1 : 0;
        }
        if((float)neighbors >= (float)pixels * 0.4)
            atomic_inc(counter);
    }
}