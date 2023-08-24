// #include <smmintrin.h> // For _mm_stream_load_si128
// #include <emmintrin.h> // For _mm_mul_ps
#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
const int VECTOR_SIZE = 8;

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
    __m256 scaleVec = _mm256_set1_ps(scale);

    for (int i = 0; i < N; i += VECTOR_SIZE) {
        __m256 xVec = _mm256_load_ps(&X[i]);
        __m256 yVec = _mm256_load_ps(&Y[i]);

        __m256 resultVec = _mm256_fmadd_ps(scaleVec, xVec, yVec);

        _mm256_stream_ps(&result[i], resultVec);
    }
}
