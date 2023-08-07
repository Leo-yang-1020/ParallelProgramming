#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;


void absSerial(float* values, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	if (x < 0) {
	    output[i] = -x;
	} else {
	    output[i] = x;
	}
    }
}

// implementation of absolute value using 15418 instrinsics
void absVector(float* values, float* output, int N) {
    __cmu418_vec_float x;
    __cmu418_vec_float result;
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

	// All ones
	maskAll = _cmu418_init_ones();

	// All zeros
	maskIsNegative = _cmu418_init_ones(0);

	// Load vector of values from contiguous memory addresses
	_cmu418_vload_float(x, values+i, maskAll);              // x = values[i];

	// Set mask according to predicate
	_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

	// Execute instruction using mask ("if" clause)
	_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

	// Inverse maskIsNegative to generate "else" mask
	maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

	// Execute instruction ("else" clause)
	_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

	// Write results back to memory
	_cmu418_vstore_float(output+i, result, maskAll);
    }
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	float result = 1.f;
	int y = exponents[i];
	float xpower = x;
	while (y > 0) {
	    if (y & 0x1)  /* Is y odd */
		result *= xpower;
	    xpower = xpower * xpower;
	    y >>= 1;
	}
	if (result > 4.18f) {
	    result = 4.18f;
	}
	output[i] = result;
    }
}

__cmu418_vec_float expBySqur(float* values, int* exponents, int idx, __cmu418_mask maskRemaining)
{
	__cmu418_vec_float x;
	__cmu418_vec_int y;
	__cmu418_vec_float result;
    __cmu418_vec_float max_result = _cmu418_vset_float(4.18f);
    __cmu418_vec_int zero = _cmu418_vset_int(0);
	__cmu418_vec_int one = _cmu418_vset_int(1);
	__cmu418_vec_int andResult;
	int bits_cnt, remainingBits;
    __cmu418_mask maskAll, maskIsOdd, maskMax;
	__cmu418_mask maskIsPositive; /* Used for judging which lane hasn't finished in one loop */

	// All ones
	maskAll = _cmu418_init_ones();

	_cmu418_vset_float(result, 1.f, maskRemaining);

	// All zeros
	maskIsOdd = _cmu418_init_ones(0);
	maskIsPositive = _cmu418_init_ones(0);
	maskMax = _cmu418_init_ones(0);

	_cmu418_vload_float(x, values+idx, maskRemaining); /* Read values */
	_cmu418_vload_int(y, exponents+idx, maskRemaining); /* Read exp */


	_cmu418_vlt_int(maskIsPositive, zero, y, maskRemaining); /* Update maskIsPotive */
	bits_cnt = _cmu418_cntbits(maskIsPositive);

	/* Doing loop utill all data finished */
	while (bits_cnt > 0) {
		_cmu418_vbitand_int(andResult, one, y, maskIsPositive); /* y & 0x1 */
		_cmu418_veq_int(maskIsOdd, one, andResult, maskIsPositive); /* if (y & 0x1) */
		maskIsOdd = _cmu418_mask_and(maskIsOdd, maskIsPositive);
		_cmu418_vmult_float(result, x, result, maskIsOdd); /* result *= xpower */

		_cmu418_vmult_float(x, x, x, maskIsPositive);  /* xpower = xpower * xpower */
		_cmu418_vshiftright_int(y, y, one, maskIsPositive); /* y >> 1 */

		_cmu418_vlt_int(maskIsPositive, zero, y, maskRemaining); /* Update maskIsPotive, check the finished data */
		bits_cnt = _cmu418_cntbits(maskIsPositive); /* Update current num of positive y */
	}
	_cmu418_vlt_float(maskMax, max_result, result, maskRemaining); /* if (result > 4.18f )*/
	_cmu418_vmove_float(result, max_result, maskMax); /* result = 4.18f */
	//addUserLog("each loop\n");

	return result;
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {
    // Implement your vectorized version of clampedExpSerial here
    //  ...
	int i = 0, remainingBits;
	__cmu418_vec_float result;
	__cmu418_mask maskRemaining; /* Used for remaining */
	__cmu418_mask maskAll; /* All 1 */

	// All ones
	maskAll = _cmu418_init_ones();
	if (N >= VECTOR_WIDTH) {
		for (i = 0; i < N; i+=VECTOR_WIDTH) {
			if (N - i < VECTOR_WIDTH)
				break;
			result = expBySqur(values, exponents, i, maskAll);
			_cmu418_vstore_float(output+i, result, maskAll);
		}
	}
	else {
		remainingBits = N;
		maskRemaining =  _cmu418_init_ones(remainingBits);
		result = expBySqur(values, exponents, 0, maskRemaining);
		_cmu418_vstore_float(output, result, maskRemaining);
		return;
	}

	/* When N is not multiple of VECTOR_WIDTH, handle remaining elements */
	if (N % VECTOR_WIDTH != 0) {
		remainingBits = N - i;
		printf("remainingBits is %d\n", remainingBits);
		maskRemaining =  _cmu418_init_ones(remainingBits);
		result = expBySqur(values, exponents, i, maskRemaining);
		_cmu418_vstore_float(output+i, result, maskRemaining);
	}

}



float arraySumSerial(float* values, int N) {
    float sum = 0;
    for (int i=0; i<N; i++) {
	sum += values[i];
    }

    return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
    // Implement your vectorized version here
    //  ...
	int i;
	__cmu418_mask maskAll;
	float result = 0;
	float result_arr[VECTOR_WIDTH];
	__cmu418_vec_float val;
	__cmu418_vec_float total_val;

	total_val = _cmu418_vset_float(0);
	maskAll = _cmu418_init_ones();

	for (i = 0; i < N; i+=VECTOR_WIDTH) {
		_cmu418_vload_float(val, values + i, maskAll);
		_cmu418_vadd_float(total_val, total_val, val, maskAll);
	}

	_cmu418_vstore_float(result_arr, total_val, maskAll);

	for (i=0; i < VECTOR_WIDTH; i++)
		result += result_arr[i];
	return result;

}
