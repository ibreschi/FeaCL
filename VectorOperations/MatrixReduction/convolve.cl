__kernel void convolve(
    const __global  float *  input,
    __constant float *   mask,
    __global  float *  output,
     int inputWidth, int maskWidth)

    {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    float sum = 0.0f;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;
        for (int c = 0; c < maskWidth; c++)
        {
            sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];
        }
}
    output[y * get_global_size(0) + x] = sum;
}