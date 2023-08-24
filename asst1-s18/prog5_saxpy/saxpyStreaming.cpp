// #include <smmintrin.h> // For _mm_stream_load_si128
// #include <emmintrin.h> // For _mm_mul_ps
#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
const int VECTOR_SIZE = 4;

extern void saxpySerial(int N,
			float scale,
			float X[],
			float Y[],
			float result[]);


void saxpyStreaming(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    // Replace this code with ones that make use of the streaming instructions
    // saxpySerial(N, scale, X, Y, result);
    __m128 vec_x, vec_y;
    // scale, scale, scale, scale
    __m128 scalar = _mm_set1_ps(scale);
    for (int i = 0; i < N; i += VECTOR_SIZE) {
        vec_x = _mm_loadu_ps(X + i);  /* load x */
        vec_y = _mm_loadu_ps(Y + i);  /* load y */
        vec_x = _mm_mul_ps(vec_x, scalar); /* multiple scalar */
        vec_x = _mm_add_ps(vec_x, vec_y); /* vector add */
        _mm_stream_ps(result + i, vec_x); /* streaming store without cache coherence */
    }
}
