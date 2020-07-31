// #include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cassert>
#include <cstdint>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#pragma STDC FENV_ACCESS ON
#include <cfenv>
#include "common.h"



#define SIGN_OFFSET_FP8       7
#define EXPONENT_OFFSET_FP8   2
#define EXPONENT_MASK_FP8     0x7C
#define EXPONENT_BIAS_FP8	  15
#define SIGNIFICAND_MASK_FP8  0x03
#define FP8_MAX			      57344
#define SIGN_OFFSET_FP32      31
#define EXPONENT_OFFSET_FP32  23
#define EXPONENT_BIAS_FP32    127
#define SIGNIFICAND_MASK_FP32 0x007FFFFF
#define EXPONENT_MASK_FP32    0x7F800000
#define SIGN_MASK_FP32        0x80000000
#define SIGN_MASK_FP8 0x80

#define SIGN_OFFSET_FP16      15
#define SIGN_MASK_FP16        0x8000
#define EXPONENT_OFFSET_FP16  10
#define EXPONENT_MASK_FP16    0x7C00
#define EXPONENT_BIAS_FP16    15
#define SIGNIFICAND_MASK_FP16 0x03FF

#define VPE_RM_NEAREST_EVEN      0
#define VPE_RM_TO_0              1
#define VPE_RM_INF               2
#define VPE_RM_NINF              3
#define VPE_RM_STOCHASTIC        4
#define VPE_RM_DEFAULT	         5
#define VPE_RM_RHAZ		 6


__device__ int lzcnt(uint32_t bits, uint32_t int_num)
{
    int msb = bits - 1;
    int lsb = 0;
    int i = msb;
    for ( ; i >= lsb; --i) {
        if ((int_num & (1 << i)) != 0) {
            break;
        }
    }
    return bits - i - 1;
}


//sbs implements select bits x[high:low]
__device__ uint32_t sbs(uint32_t x, uint8_t high, uint8_t low)
{
  return (high==31) ? (x>>low) : ((x&((1<<(high+1)) - 1))>>low);
}
//cbs implements concatenate bits {x[31-pos:0],y[pos-1,0]}
__device__ uint32_t cbs(uint32_t x, uint32_t y, uint8_t pos)
{
  return ((x<<pos) | (y&((1<<pos) - 1)));
}
//ibs implements insert bits x[high:low] = y[high-low-1:0]
__device__ uint32_t ibs(uint32_t x, uint32_t high, uint32_t low, uint32_t y)
{
  return (high==31) ? ((x&((1<<low)-1)) | (y<<low)) : ((x&(~((1<<(high+1)) - (1<<low)))) | ((y<<low)&(((1<<(high+1)) - 1))));
}

__device__ int fp_accommodate_rounding( uint32_t intValuePreRounding
                                    , bool roundedMSB, bool roundedLSBs
                                    , unsigned int sign, int roundingMode
                                    , uint32_t lfsrVal, uint32_t discardedAlignedLeft )
{
	uint32_t  result = 0;
	result = intValuePreRounding;
	switch (roundingMode)
	{
	case VPE_RM_TO_0:
		result = intValuePreRounding;
		break;
	case VPE_RM_INF:
		if ((sign == 0) && ((roundedMSB == 1) || (roundedLSBs == 1)))
		{
			result = intValuePreRounding + 1;
		}
		break;
	case VPE_RM_NINF:
		if	((sign == 1) && ((roundedMSB == 1) || (roundedLSBs == 1)))
		{
			result = intValuePreRounding + 1;
		}
		break;
	case VPE_RM_RHAZ:
		if (roundedMSB == 1) //half or above half will be rounded away from zero
		{
			result = intValuePreRounding + 1;
		}
		break;
	case VPE_RM_STOCHASTIC:
		if(discardedAlignedLeft >= lfsrVal)
		{
			result = intValuePreRounding + 1;
		}
		break;
	case VPE_RM_NEAREST_EVEN:
	default:
		if ((((intValuePreRounding & 0x1) == 1) && (roundedMSB == 1)) ||
			(((intValuePreRounding & 0x1) == 0) && (roundedMSB == 1) && (roundedLSBs == 1)))
		{
			result = intValuePreRounding + 1;
		}
		break;
	}
	return result;
}


__device__ int fp_accommodate_rounding_fp8(uint32_t intValuePreRounding, bool roundedMSB, bool roundedLSBs, int32_t roundedBits, unsigned int sign, float random, int roundingMode)
{
	uint32_t  result = 0;
	result = intValuePreRounding;
	if (roundingMode == 0) // RNE
	{
		if ((((intValuePreRounding & 0x1) == 1) && (roundedMSB == 1)) ||
			(((intValuePreRounding & 0x1) == 0) && (roundedMSB == 1) && (roundedLSBs == 1)))
		{
			result = intValuePreRounding + 1;
		}
	}
	else if (roundingMode == 1) // Stochastic
	{
		int adjusted_rnd = random * ((1<<21)-1);
		//printf("%d %d %d\n", adjusted_rnd, roundedBits, (roundedBits >= adjusted_rnd));

		if (roundedBits >= adjusted_rnd)
		{
			result = intValuePreRounding + 1;
		}

	}
	return result;
}


__device__ bool fp32_is_zero(uint32_t val)
{
    return (val & (~SIGN_MASK_FP32)) ? 0 : 1;
}

__device__ bool fp32_is_infinity(uint32_t val)
{
    return (val & 0x7FFFFFFF) == 0x7F800000 ? 1 : 0;
}

__device__ bool fp32_is_nan(uint32_t val)
{
    bool isAllExponentBitsSet = ((val & 0x7f800000) == 0x7f800000);
    bool isAnyMantissaBitSet = ((val & 0x007fffff) != 0);
    return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

__device__ bool fp32_is_denormal(uint32_t val)
{
    return (((val & 0x7f800000) == 0) && ((val & 0x007fffff) != 0));
}

__device__ bool fp32_is_negative(uint32_t val)
{
    return (val & SIGN_MASK_FP32) != 0;
}

__device__ float bf16_to_fp32(uint16_t input)
{
	uint32_t uintRes = input << 16;
	return *(float*)&uintRes;
}


// Rounding mode: 0 - round down (trim), 1 - round half away from zero, 2 - stochastic, 3 - RNE
__device__ uint16_t fp32_to_bf16(float input, int roundingMode, uint32_t lfsrVal)
{
	const uint32_t &inputUint = *(const uint32_t *)&input;

	uint16_t res;

	if (fp32_is_nan(inputUint))
	{
		res = 0x7fc0;
	}
	else
	{
		uint32_t inputSign = (inputUint & (1UL << 31)) >> 31;
		bool roundedMSB = ((inputUint & (1<<15)) != 0);
		bool roundedLSB = ((inputUint & ((1<<15) - 1)) != 0);

		int32_t inputExponent = (inputUint >> EXPONENT_OFFSET_FP32) & 0xFF;

		int32_t outputExponent = inputExponent;

		uint32_t inputMantissa = inputUint & ((1 << (EXPONENT_OFFSET_FP32+1)) - 1);
		inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

		int32_t outputMantissa = inputMantissa >> 16;

		if (roundingMode == 1 && roundedMSB)
		{
			outputMantissa++;
		}
		else if (roundingMode == 2)
		{
		    uint32_t trimmedBits = inputUint << 16;
		    if (trimmedBits > lfsrVal)
		    {
		        outputMantissa++;
		    }
		}
		else if (roundingMode == 3)
		{
		    bool lsb =  ((inputUint & (1<<16)) != 0);
		    if (lsb)
		    { // The mantissa is odd - round up if roundedMSB is true
		        outputMantissa = (roundedMSB) ? (outputMantissa+1) : outputMantissa;
		    }
		    else
		    {
		        // The mantissa is even - round up only if both roundedMSB and roundedLSB are true
		        outputMantissa = (roundedMSB && roundedLSB) ? (outputMantissa+1) : outputMantissa;
		    }
		}
		if (outputMantissa & (1 << 8))
		{
			outputExponent++;
		}

		res = (inputSign << 15) | (outputExponent << 7) | (outputMantissa & 0x7F);

	}

	return res;

}




__device__ bool fp16_is_zero(uint16_t val)
{
	return (val & (~SIGN_MASK_FP16)) ? 0 : 1;
}

__device__ bool fp16_is_infinity(uint16_t val)
{
	return (val & 0x7FFF) == EXPONENT_MASK_FP16 ? 1 : 0;
}

__device__ bool fp16_is_nan(uint16_t val)
{
	bool isAllExponentBitsSet = ((val & EXPONENT_MASK_FP16) == EXPONENT_MASK_FP16);
	bool isAnyMantissaBitSet = ((val & SIGNIFICAND_MASK_FP16) != 0);
	return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

__device__ bool fp16_is_negative(uint16_t val)
{
	return ((val & SIGN_MASK_FP16) == SIGN_MASK_FP16);
}

__device__ bool fp16_is_denormal(uint16_t val)
{ // Do not consider zero as denormal
	return (((val & EXPONENT_MASK_FP16) == 0) && ((val & SIGNIFICAND_MASK_FP16) != 0));
}



__device__ void fp16_to_fp32(uint16_t input, float *output)
{
	const uint16_t inputUint = input;
	uint32_t *outputUint = (uint32_t *)output;

	int32_t inputMantissa = (inputUint & SIGNIFICAND_MASK_FP16);
	int32_t inputExponent = (inputUint & EXPONENT_MASK_FP16) >> EXPONENT_OFFSET_FP16;
	int32_t inputSign = (inputUint & SIGN_MASK_FP16) >> SIGN_OFFSET_FP16;

	int32_t outputExponent;
	int32_t outputMantissa;
	int32_t outputSign = inputSign;

	if (fp16_is_zero(inputUint))
	{
		outputExponent = 0x0;
		outputMantissa = 0x0;
	}
	else if (fp16_is_nan(inputUint))
	{
		outputExponent = 0xFF;
		outputMantissa = 0x007FFFFF;
		outputSign = 0;
	}
	else if (fp16_is_infinity(inputUint))
	{
		outputExponent = 0xFF;
		outputMantissa = 0x0;
	}
	else
	{
		outputExponent = inputExponent - EXPONENT_BIAS_FP16 + EXPONENT_BIAS_FP32;
		int32_t mantissaForAdjustment = inputMantissa;
		if (fp16_is_denormal(inputUint))
		{
			int shift = lzcnt(EXPONENT_OFFSET_FP16, inputMantissa);
			// Shift leading 1 to bit 10 (normalize) and fixup the exponent accordingly
			mantissaForAdjustment = (inputMantissa << (shift + 1)) & SIGNIFICAND_MASK_FP16;
			outputExponent -= shift;
		}
		// Normal case
		outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16);
	}

	*outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

}




__device__ void fp32_to_fp16(float input, uint16_t *output, int roundingMode, int32_t lfsrVal)
{
	int inputExponent, inputSign, unbiasedExp = 0;
	uint32_t inputMantissa;
	bool roundedMSB = 0, roundedLSBs = 0;
	int minExp = -25;
	int minNormExp = -14;
	int maxExp = 15;

	const uint32_t inputUint = *(const uint32_t *)&input;

	inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
	inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
	inputSign = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

	int32_t outputExponent;
	int32_t outputMantissa;
	int32_t outputSign = inputSign;
	if (fp32_is_nan(inputUint))
	{
		// return the same NAN always (0x7FFF), as NVDA does 
		outputSign = 0x0;
		outputExponent = 0x1F;
		outputMantissa = 0x3FF;
	}
	else if (fp32_is_zero(inputUint))
	{
		// return +-0
		outputExponent = 0x0;
		outputMantissa = 0x0;
	}
	else if (fp32_is_infinity(inputUint))
	{
		// return +-infinity
		outputExponent = 0x1F;
		outputMantissa = 0x0;
	}
	else
	{
		// Valid number
		unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
		inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

		if (unbiasedExp > maxExp)
		{

			if ((roundingMode == (VPE_RM_TO_0)) ||
				(inputSign && (roundingMode == (VPE_RM_INF))) ||
				(!inputSign && (roundingMode == (VPE_RM_NINF)))
				)

			{ // +- 65504.0 - that's what NVDA does
				outputMantissa = 0x3FF;
				outputExponent = maxExp + EXPONENT_BIAS_FP16;
			}
			else
			{ // +-infinity
				outputExponent = 0x1F;
				outputMantissa = 0x0;
			}
		}
		else if (unbiasedExp < minExp)
		{
			// The result will be either 0 or 0x1
			roundedMSB = 0;
			roundedLSBs = 1;
			outputMantissa = fp_accommodate_rounding(0, roundedMSB, roundedLSBs, inputSign, roundingMode, lfsrVal, 0);
			outputExponent = 0x0;
		}
		else
		{ // minExp <= unbiasedExp <= maxExp
			outputExponent = unbiasedExp;
			int rc_bit_idx = (unbiasedExp < minNormExp) ? -(unbiasedExp + 2) : (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16 - 1);
			roundedMSB = ((inputMantissa >> rc_bit_idx) & 0x1) != 0;
			roundedLSBs = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
			uint32_t discardedAlignedLeft = inputMantissa << (31 - rc_bit_idx);
			outputMantissa = inputMantissa >> (rc_bit_idx + 1);
			outputMantissa = fp_accommodate_rounding(outputMantissa, roundedMSB, roundedLSBs, inputSign, roundingMode, lfsrVal, discardedAlignedLeft);
			if (((unbiasedExp < minNormExp) && (outputMantissa & (1 << EXPONENT_OFFSET_FP16))) || (outputMantissa & (1 << (EXPONENT_OFFSET_FP16 + 1))))
			{ // Should handle two cases: 
			  // 1. The number was denormal, and after rounding became normal
			  // 2. The number was rounded to the 1.0 * 2^(next exponent)
				outputExponent = outputExponent + 1;
			}
			if (outputExponent > maxExp)
			{
				// return infinity
				outputExponent = 0x1F;
				outputMantissa = 0x0;
			}
			else
			{
				if (outputExponent < minNormExp)
				{
					outputExponent = 0x0;
				}
				else
				{
					outputExponent += EXPONENT_BIAS_FP16;
				}
				// normalize - leave 10 bits
				outputMantissa &= SIGNIFICAND_MASK_FP16;
			}

		}
	}
	*output = outputMantissa | (outputExponent << EXPONENT_OFFSET_FP16) | (outputSign << SIGN_OFFSET_FP16);

}

//default values exp_width=5, man_width=2, exp_bias=7
//man_width must be at least 1
//sign is always 1 bit (upper bit)
// Rounding mode: 0 - RNE, 1 - STOCHASTIC
__device__ void fp32_to_fp8(float input, uint8_t *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias, int roundingMode, int32_t lfsrVal)
{
	int inputExponent, inputSign, unbiasedExp = 0;
	uint32_t inputMantissa;
	bool roundedMSB = 0, roundedLSBs = 0;
	//int minExp = -25;
	int minNormExp = 1 - exp_bias; //-14
	int maxExp = ((1 << exp_width) - 1) - exp_bias - 1; //15
	int minExp = minNormExp - man_width - 1; //-25 //min denormal value can come from rounding of 0.5
	int32_t exponent_offset_fp8 = man_width;
	int32_t sign_offset_fp8 = 7;

	const uint32_t inputUint = *(const uint32_t *)&input;

	inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
	inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
	inputSign = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

	int32_t outputExponent;
	int32_t outputMantissa;
	int32_t outputSign = inputSign;
	if (fp32_is_nan(inputUint))
	{
		// return the same NAN always (0x7F)
		outputSign = 0x0;
		outputExponent = sbs(0xff,exp_width-1,0);//0x1F;
		outputMantissa = sbs(0xff,man_width-1,0);//0x3;
	}
	else if (fp32_is_zero(inputUint))
	{
		// return +-0
		outputExponent = 0x0;
		outputMantissa = 0x0;
	}
	else if (fp32_is_infinity(inputUint))
	{
		// return +-infinity
		outputExponent = sbs(0xff,exp_width-1,0);//0x1F;
		outputMantissa = 0x0;
	}
	else
	{
		// Valid number
		unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
		inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

		if (unbiasedExp > maxExp)
		{

			if ((roundingMode == (VPE_RM_TO_0)) ||
				(inputSign && (roundingMode == (VPE_RM_INF))) ||
				(!inputSign && (roundingMode == (VPE_RM_NINF)))
				)

			{ // +- max_normal
				outputMantissa = sbs(0xff,man_width-1,0);//0x3;
				outputExponent = maxExp + exp_bias;
			}
			else
			{ // +-infinity
				outputExponent = sbs(0xff,exp_width-1,0);//0x1F;
				outputMantissa = 0x0;
			}
		}
		else if (unbiasedExp < minExp)
		{
			// The result will be either 0 or 0x1
			roundedMSB = 0;
			roundedLSBs = 1;
			outputMantissa = fp_accommodate_rounding(0, roundedMSB, roundedLSBs, inputSign, roundingMode, lfsrVal, 0);
			outputExponent = 0x0;
		}
		else
		{ // minExp <= unbiasedExp <= maxExp
			outputExponent = unbiasedExp;
			int rc_bit_idx = (unbiasedExp < minNormExp) ? ((EXPONENT_OFFSET_FP32 - exponent_offset_fp8 - 1) + (minNormExp - unbiasedExp)) : (EXPONENT_OFFSET_FP32 - exponent_offset_fp8 - 1);
			roundedMSB = (((inputMantissa >> rc_bit_idx)) & 0x1) != 0;
			roundedLSBs = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
			uint32_t discardedAlignedLeft = inputMantissa << (31 - rc_bit_idx);
			outputMantissa = inputMantissa >> (rc_bit_idx + 1);
			outputMantissa = fp_accommodate_rounding(outputMantissa, roundedMSB, roundedLSBs, inputSign, roundingMode, lfsrVal, discardedAlignedLeft);
			if (((unbiasedExp < minNormExp) && (outputMantissa & (1 << exponent_offset_fp8))) || (outputMantissa & (1 << (exponent_offset_fp8 + 1))))
			{ // Should handle two cases: 
			  // 1. The number was denormal, and after rounding became normal
			  // 2. The number was rounded to the 1.0 * 2^(next exponent)
				outputExponent = outputExponent + 1;
			}
			if (outputExponent > maxExp)
			{
				// return infinity
				outputExponent = sbs(0xff,exp_width-1,0);//0x1F;
				outputMantissa = 0x0;
			}
			else
			{
				if (outputExponent < minNormExp)
				{
					outputExponent = 0x0;
				}
				else
				{
					outputExponent += exp_bias;
				}
				// normalize - leave man_width bits
				outputMantissa = sbs(outputMantissa, man_width-1, 0);
			}

		}
	}
	*output = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);

}

__device__ bool fp8_is_zero(uint8_t val)
{
	return (val & (~SIGN_MASK_FP8)) ? 0 : 1;
}

__device__ bool fp8_is_infinity(uint8_t val, uint8_t exponent_offset_fp8)
{
	bool isAllExponentBitsSet = sbs(val,6,exponent_offset_fp8) == sbs(0xff,6,exponent_offset_fp8);
	bool isAllMantissaBitsZero = (sbs(val,exponent_offset_fp8-1,0) == 0);
	return (isAllExponentBitsSet & isAllMantissaBitsZero);
}

__device__ bool fp8_is_nan(uint8_t val, uint8_t exponent_offset_fp8)
{
	bool isAllExponentBitsSet = sbs(val,6,exponent_offset_fp8) == sbs(0xff,6,exponent_offset_fp8);
	bool isAnyMantissaBitSet = (sbs(val,exponent_offset_fp8-1,0) != 0);
	return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

__device__ bool fp8_is_negative(uint8_t val)
{
	return ((val & SIGN_MASK_FP8) == SIGN_MASK_FP8);
}


__device__ bool fp8_is_denormal(uint8_t val, uint8_t exponent_offset_fp8)
{ // Do not consider zero as denormal
	bool isAllExponentBitsZero = sbs(val,6,exponent_offset_fp8) == 0;
	bool isAnyMantissaBitSet = (sbs(val,exponent_offset_fp8-1,0) != 0);
	return (isAllExponentBitsZero & isAnyMantissaBitSet);
}


//default values exp_width=5, man_width=2, exp_bias=7
//man_width must be at least 1
//sign is always 1 bit (upper bit)
__device__ void fp8_to_fp32(uint8_t input, float *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias)
{
	const uint8_t inputUint = input;
	uint32_t *outputUint = (uint32_t *)output;
	int32_t exponent_offset_fp8 = man_width;
	int32_t sign_offset_fp8 = 7;

	int32_t inputMantissa = sbs(inputUint,man_width-1,0);
	int32_t inputExponent = sbs(inputUint,6,exponent_offset_fp8);
	int32_t inputSign = sbs(inputUint,sign_offset_fp8,sign_offset_fp8);

	int32_t outputExponent;
	int32_t outputMantissa;
	int32_t outputSign = inputSign;

	if (fp8_is_zero(inputUint))
	{
		outputExponent = 0x0;
		outputMantissa = 0x0;
	}
	else if (fp8_is_nan(inputUint, exponent_offset_fp8))
	{
		outputExponent = 0xFF;
		outputMantissa = 0x007FFFFF;
		outputSign = 0;
	}
	else if (fp8_is_infinity(inputUint, exponent_offset_fp8))
	{
		outputExponent = 0xFF;
		outputMantissa = 0x0;
	}
	else
	{
		outputExponent = inputExponent - exp_bias + EXPONENT_BIAS_FP32;
		int32_t mantissaForAdjustment = inputMantissa;
		if (fp8_is_denormal(inputUint, exponent_offset_fp8))
		{
			int shift = lzcnt(exponent_offset_fp8, inputMantissa);
			// Shift leading 1 (normalize) and fixup the exponent accordingly
			mantissaForAdjustment = sbs((inputMantissa << (shift + 1)),man_width-1,0);
			outputExponent -= shift;
		}
		// Normal case
		outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP32 - exponent_offset_fp8);
	}

	*outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

}

// fp32_to_fp8(float input, uint8_t *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias, int roundingMode, int32_t lfsrVal)
// __device__ void fp8_to_fp32(uint8_t input, float *output, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias)

__global__ void ConvertFP32FP8Kernel(const float* in_data, uint8_t* out_data, const int totalElements, const uint8_t exp_width, const uint8_t man_width, const uint8_t exp_bias, const int roundingMode, const int32_t lfsrVal)
{
	CUDA_KERNEL_LOOP(i, totalElements){
		fp32_to_fp8(in_data[i], &out_data[i], exp_width, man_width, exp_bias, roundingMode, lfsrVal);
	}
}

__global__ void ConvertFP8FP32Kernel(const uint8_t* in_data, float* out_data, const int totalElements, const uint8_t exp_width, const uint8_t man_width, const uint8_t exp_bias)
{
	CUDA_KERNEL_LOOP(i, totalElements){
		fp8_to_fp32(in_data[i], &out_data[i], exp_width, man_width, exp_bias);
	}
}


__global__ void TruncFP8Kernel(const float* in_data, float* out_data, const int totalElements, const uint8_t exp_width, const uint8_t man_width, const uint8_t exp_bias, const int roundingMode, const int32_t lfsrVal)
{
	CUDA_KERNEL_LOOP(i, totalElements){
		uint8_t out_fp8;
		fp32_to_fp8(in_data[i], &out_fp8, exp_width, man_width, exp_bias, roundingMode, lfsrVal);
		fp8_to_fp32(out_fp8, &out_data[i], exp_width, man_width, exp_bias);
	}
}


__global__ void TruncBF16Kernel(const float* in_data, float* out_data, const int totalElements, const int roundingMode) 
{
	CUDA_KERNEL_LOOP(i, totalElements){
      out_data[i] = bf16_to_fp32(fp32_to_bf16(in_data[i], roundingMode, 0));
	}

}

torch::Tensor trunc_bf16_cuda(torch::Tensor input, const bool inplace, const int roundingMode) {
  const auto num_elements = input.numel();
  torch::Tensor output;
  if (inplace)
    output = input;
  else
    output = torch::empty_like(input);

  TruncBF16Kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(input.data<float>(), output.data<float>(), num_elements, roundingMode);
  return output;
}

torch::Tensor trunc_fp8_cuda(torch::Tensor input, const bool inplace, const int exp_width, const int man_width, const int exp_bias, const int roundingMode, const int lfsrVal) {
	const auto num_elements = input.numel();
	torch::Tensor output;
	if (inplace)
	  output = input;
	else
	  output = torch::empty_like(input);
	torch::Tensor rand = torch::rand_like(input);
	TruncFP8Kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(input.data<float>(), output.data<float>(), num_elements, exp_width, man_width, exp_bias, roundingMode, lfsrVal);
	return output;
  }

  
torch::Tensor fp32_to_fp8_cuda(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias, const int roundingMode, const int lfsrVal) {
	const auto num_elements = input.numel();
	torch::Tensor output = torch::empty_like(input, torch::dtype(torch::kUInt8));
	torch::Tensor rand = torch::rand_like(input);
	ConvertFP32FP8Kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(input.data<float>(), output.data<uint8_t>(), num_elements, exp_width, man_width, exp_bias, roundingMode, lfsrVal);
	return output;
}

torch::Tensor fp8_to_fp32_cuda(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias) {
	const auto num_elements = input.numel();
	torch::Tensor output = torch::empty_like(input, torch::dtype(torch::kFloat32));
  
	ConvertFP8FP32Kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(input.data<uint8_t>(), output.data<float>(), num_elements, exp_width, man_width, exp_bias);
	return output;
}
