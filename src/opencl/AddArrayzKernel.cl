__kernel void add_floats(__global float* a, __global float* b, __global float* out, int n)
{
    int i = get_global_id(0);
    if( i >= n ){
        return;
    }
    for(int j=0; j<8192*32; j++){
        a[i] = cos((float)i);
        b[i] = sin((float)i);
        out[i] += a[i] + b[i];
    }
}

