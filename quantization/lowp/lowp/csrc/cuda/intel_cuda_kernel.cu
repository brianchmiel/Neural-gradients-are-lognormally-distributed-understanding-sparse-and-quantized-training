#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>


#define CUBLOCK_SIZE 1024 
enum ROUNDING_MODES{TRUNCATE=0, ROUND_RNE=1, ROUND_STOCHASTIC=2, ROUND_RNAZ=3, ROUND_RNTZ=4};

namespace {

typedef union half_t { 
   unsigned short u; 
   torch::Half f; 
} __half_t; 

typedef union ufloat32
{
  unsigned u;
  float f;
}__float_t;


__device__ __forceinline__ uint32_t rotl_(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}

__device__ static uint32_t  s[4] = {0x76B5DBC3, 0x532CB7BF, 0x6AFA41C3, 0x28DBD9F7};

__device__ __forceinline__ uint32_t _xorshf_rand(void) {
	const uint32_t result_plus = s[0] + s[3];
	const uint32_t t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl_(s[3], 11);

	return result_plus;
}

__device__ static uint32_t  s1[4] = {1387366120, 279844183, 888998500, 1099633400}; 
__device__ static uint32_t  s2[4] = {2034269327, 2125325156, 1209715489, 193165672};
__device__ static uint32_t  s3[4] = {1555452618, 650181557, 883695203, 62767784};
__device__ static uint32_t  s4[4] = {419524804, 2146478152, 480059239, 1468956197};
__device__ static uint32_t  s5[4] = {1252084877, 500390994, 977516591, 1950666000}; 
__device__ static uint32_t  s6[4] = {393659750, 834151069, 1477014702, 734008143};
__device__ static uint32_t  s7[4] = {1983400973, 116410309, 2110188261, 2019272068}; 
__device__ static uint32_t  s8[4] = {187709636, 28336299, 419632041, 1774181187}; 
__device__ static uint32_t  s9[4] = {702309618, 407781555, 1512057936, 1868769368}; 
__device__ static uint32_t  s10[4] = {510001215, 966559856, 776583255, 147562106};
__device__ static uint32_t  s11[4] = {127180605, 1881312534, 478635452, 814821902}; 
__device__ static uint32_t  s12[4] = {733990058, 1889991804, 1108257970, 1093480892}; 
__device__ static uint32_t  s13[4] = {427374380, 416747337, 558000409, 1594848927}; 
__device__ static uint32_t  s14[4] = {444870959, 1595722866, 1064124488, 363710254}; 
__device__ static uint32_t  s15[4] = {703721499, 389640783, 1002360059, 1427395742}; 
__device__ static uint32_t  s16[4] = {1295231497, 1254972431, 1423497865, 861918264};

__device__ static uint32_t  *sptr[16] = {s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16};

__device__ __forceinline__ uint32_t _xorshf_rand_with_seed(uint32_t *ps) {
	const uint32_t result_plus = ps[0] + ps[3];
	const uint32_t t = ps[1] << 9;

	ps[2] ^= ps[0];
	ps[3] ^= ps[1];
	ps[1] ^= ps[2];
	ps[0] ^= ps[3];

	ps[2] ^= t;

	ps[3] = rotl_(ps[3], 11);

	return result_plus;
}

template <typename scalar_t>
__device__ __forceinline__ float __anyfloat2float_rn( scalar_t a_)
{
  float f_;
  if (std::is_same<scalar_t, double>::value ) {
      	f_ = __double2float_rn(a_);
  } else if (std::is_same<scalar_t, float>::value) {
        f_ = a_;
  } else if (std::is_same<scalar_t, torch::Half>::value) {
	f_ = __half2float((torch::Half)a_);
  }
  return f_;
}

template <typename scalar_t>
__device__ __forceinline__ void __float2anyfloat_rn( float f_, scalar_t *out)
{
  scalar_t a_;
  if (std::is_same<scalar_t, double>::value ) {
      	a_ = (scalar_t)(f_);
  } else if (std::is_same<scalar_t, float>::value) {
	a_ = f_;
  } else if (std::is_same<scalar_t, torch::Half>::value) {
	a_ = (torch::Half)__float2half_rn(f_);
  }
  *out =  a_;
}

template <typename scalar_t>
__device__ __forceinline__ torch::Half __anyfloat2half_rn( scalar_t f_)
{
  torch::Half h_;
  if (std::is_same<scalar_t, double>::value ) {
      	h_ = __float2half_rn(__double2float_rn(f_));
  } else if (std::is_same<scalar_t, float>::value) {
	h_ = __float2half_rn(f_);
  } else if (std::is_same<scalar_t, torch::Half>::value) {
	h_ = (torch::Half)f_;
  }
  return h_;
}

template <typename scalar_t>
__device__ __forceinline__ void __half2anyfloat(torch::Half h_, scalar_t *out)
{
  scalar_t f_;
  if (std::is_same<scalar_t, double>::value ) {
      	f_ = (scalar_t)__half2float((torch::Half)h_);
  } else if (std::is_same<scalar_t, float>::value) {
	f_ = __half2float(h_);
  } else if (std::is_same<scalar_t, torch::Half>::value) {
	f_ = (torch::Half)h_;
  }
  *out = f_;
}

template <typename scalar_t>
__global__ void QuantEmuBFloat16Kernel(
	const scalar_t *in, 
	scalar_t *out, 
	const int size, 
	int rmode)
{
  int lshift = 16;
  unsigned int mask_mant = (unsigned int)(0xFFFFFFFF << lshift);
  unsigned int grs_bitmask = 0x0000FFFF; 
  unsigned int rne_tie = 0x00018000; 

  unsigned short rne_mask = 0; /* round to nearest even mask */ 
  unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
  unsigned short rnaz_mask = 0; /* round to nearest away from zero mask */ 
  unsigned short rntz_mask = 0; /* round to nearest towards zero mask */ 
  if (rmode == ROUND_RNE) rne_mask = 1;  
  if (rmode == ROUND_RNAZ) rnaz_mask = 1;  
  if (rmode == ROUND_RNTZ) rntz_mask = 1;  
  if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __float_t uf; 
      uf.f = __anyfloat2float_rn(in[gid]); 
      unsigned int is_normal = (((uf.u & 0x7F800000) <= 0x7F000000) && ((uf.u & 0x7F800000) >= 0x00800000))?1:0;
      unsigned int is_denorm = ((uf.u & 0x7F800000) == 0x0)?1:0;
      unsigned int is_naninf = ((uf.u & 0x7F800000) == 0x7F800000)?1:0;
      /* nearest rounding masks */ 
      unsigned int rnmask = (uf.u & grs_bitmask); 
      unsigned int rnmask_tie = (uf.u & rne_tie);  
      if (is_naninf == 0 && is_normal) { 
        if (sr_mask) { 
          /* stochastic with 16 seeds */ 
          int seed_index = (gid/16); 
          unsigned int rand = _xorshf_rand_with_seed(sptr[(seed_index%16)]);
          /* apply stochastic rounding before truncation if sr_mask is enabled */ 
          uf.u += (rand & 0x0000FFFF); 
        } else { 
          /* round to nearest even, if rne_mask is enabled */ 
          uf.u += rne_mask * (((rnmask > 0x00008000) || (rnmask_tie == rne_tie)) << lshift); 
          /* round to nearest away from zero, if rnaz_mask is enabled */ 
          uf.u += rnaz_mask * ((rnmask >= 0x00008000) << lshift);  
          /* round to nearest towards zero, if rntz_mask is enabled */ 
          uf.u += rntz_mask * ((rnmask > 0x00008000) << lshift);  
        }
      } else if (is_denorm) {
        /* Flush Denormal */ 
        uf.u = 0; 
      }
       /* truncation */ 
      uf.u = (uf.u & mask_mant); 

      __float2anyfloat_rn(uf.f, &out[gid]);
  }
}

template <typename scalar_t>
__global__  void QuantEmuFloat16Kernel(
	const scalar_t *in, 
	scalar_t *out,
	const int size, 
	int no_denorm) 
{
  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      torch::Half hval;
      hval = __anyfloat2half_rn(in[gid]);
      h.f = hval;
      unsigned short not_denorm = ((((h.u & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;
      h.u *= !(is_denorm * no_denorm);
      __half2anyfloat(h.f, &out[gid]);
  }
}

template <typename scalar_t>
__global__ void QuantEmuFloat8Kernel(
	const scalar_t* __restrict__ in, 
	scalar_t* __restrict__ out, 
	const int size, 
	int mbits, 
	int exp_bits, 
	int rmode)
{
  int non_mant_bits = exp_bits + 1; /* exponent + sign */
  int lshift = 10 - (mbits - non_mant_bits);

  unsigned short rne_mask = 0; /* round to nearest even mask */ 
  unsigned short rnaz_mask = 0; /* round to nearest away from zero mask */ 
  unsigned short rntz_mask = 0; /* round to nearest towards zero mask */ 
  unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
  if (rmode == ROUND_RNE) rne_mask = 1;  
  if (rmode == ROUND_RNAZ) rnaz_mask = 1;  
  if (rmode == ROUND_RNTZ) rntz_mask = 1;  
  if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

  unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);
  unsigned short grs_bitmask = 0x00FF; 
  unsigned short rne_tie = 0x0180; 

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      float inval = in[gid];
      __half  hval = __anyfloat2half_rn(inval); 
      //__half  hval = __float2half_rn(inval); 
      h.f = hval;
      /* values above 57344.0, saturate them to +- Infinity */
      if ((h.u & 0x7FFF) > 0x7B00) h.u = ((h.u & 0x8000) | 0x7C00);
     
      unsigned short can_round = ((h.u & 0x7F00) < 0x7B00)?1:0; 
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400))?1:0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0)?1:0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00)?1:0;
      /* nearest rounding masks */ 
      unsigned short rnmask = (h.u & grs_bitmask); 
      unsigned short rnmask_tie = (h.u & rne_tie);  
      if (is_naninf == 0) { 
        if (sr_mask) { 
          /* stochastic with 16 seeds */ 
          int seed_index = (gid/16); 
          unsigned short rand = (unsigned short) _xorshf_rand_with_seed(sptr[(seed_index%16)]);
          /* apply stochastic rounding before truncation if sr_mask is enabled */ 
          h.u += can_round * is_normal * (rand & 0xFF); 
          /* stochastic round:  denormals --> rne rounding */ 
          h.u += can_round * is_denorm * (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift); 
        } else { 
          /* round to nearest even, if rne_mask is enabled */ 
          h.u += can_round * rne_mask * (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift); 
          /* round to nearest away from zero, if rnaz_mask is enabled */ 
          h.u += can_round * rnaz_mask * ((rnmask >= 0x0080) << lshift);  
          /* round to nearest towards zero, if rntz_mask is enabled */ 
          h.u += can_round * rntz_mask * ((rnmask > 0x0080) << lshift);  
        }
      }
       /* truncation */ 
      h.u = (h.u & mask_mant); 
 
      __half2anyfloat(h.f, &out[gid]);
      //float outval = __half2float(h.f);
      //out[gid] = outval;
  }
}

template <typename scalar_t>
__global__ void QuantEmuFloat8FlushDenormKernel(
	const scalar_t* __restrict__ in, 
	scalar_t* __restrict__ out, 
	const int size, 
	int mbits, 
	int exp_bits, 
	int rmode)
{
  int non_mant_bits = exp_bits + 1; /* exponent + sign */
  int lshift = 10 - (mbits - non_mant_bits);

  unsigned short rne_mask = 0; /* round to nearest even mask */ 
  unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
  unsigned short rnaz_mask = 0; /* round to nearest away from zero mask */ 
  unsigned short rntz_mask = 0; /* round to nearest towards zero mask */ 
  if (rmode == ROUND_RNE) rne_mask = 1;  
  if (rmode == ROUND_RNAZ) rnaz_mask = 1;  
  if (rmode == ROUND_RNTZ) rntz_mask = 1;  
  if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

  unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);
  unsigned short grs_bitmask = 0x00FF; 
  unsigned short rne_tie = 0x0180; 

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      float inval = in[gid];
      __half  hval = __anyfloat2half_rn(inval); 
      h.f = hval;
      /* values above 57344.0, saturate them to +- Infinity */
      if ((h.u & 0x7FFF) > 0x7B00) h.u = ((h.u & 0x8000) | 0x7C00);
   
      unsigned short can_round = ((h.u & 0x7F00) < 0x7B00)?1:0; 
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400))?1:0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0)?1:0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00)?1:0;
      /* nearest rounding masks */ 
      unsigned short rnmask = (h.u & grs_bitmask); 
      unsigned short rnmask_tie = (h.u & rne_tie);  
      if (is_naninf == 0 && is_normal ) { 
        if (sr_mask) { 
          /* stochastic with 16 seeds */ 
          int seed_index = (gid/16); 
          unsigned short rand = (unsigned short) _xorshf_rand_with_seed(sptr[(seed_index%16)]);
          /* apply stochastic rounding before truncation if sr_mask is enabled */ 
          h.u += can_round * (rand & 0xFF); 
        } else { 
          /* round to nearest even, if rne_mask is enabled */ 
          h.u += can_round * rne_mask * (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift); 
          /* round to nearest away from zero, if rnaz_mask is enabled */ 
          h.u += can_round * rnaz_mask * ((rnmask >= 0x0080) << lshift);  
          /* round to nearest towards zero, if rntz_mask is enabled */ 
          h.u += can_round * rntz_mask * ((rnmask > 0x0080) << lshift);  
        }
      } else if (is_denorm) {
        /* Flush Denormal */ 
        h.u = 0; 
      }
       /* truncation */ 
      h.u = (h.u & mask_mant); 
 
      __half2anyfloat(h.f, &out[gid]);
  }
}

} // namespace

torch::Tensor quantemu_cuda_forward(
    torch::Tensor input,
    std::string mode,
    bool inplace) {
  const auto size = input.numel();
  auto sizes = input.sizes(); 
  //std::cout << "sizes from cuda : " << sizes << ", and size : " << size << ", mode: " << mode << ", nbits :" << nbits <<  std::endl; 
#if 0
  torch::Tensor output; 
  //if (inplace ) output.set_data(input); 
  if (inplace ) output.set_(input); 
  else output = torch::zeros_like(input); 
#endif 
  torch::Tensor output; 
  if (!inplace ) output = torch::zeros_like(input); 

  const int threads = CUBLOCK_SIZE;
  const dim3 blocks(( size + (CUBLOCK_SIZE-1))/CUBLOCK_SIZE); 

  if (!mode.compare("FLOAT8_RNE")) {
 	  //std::cout <<"FLOAT8 RNE called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuFloat8RNEKernel", ([&] {
    		QuantEmuFloat8Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
        	8,
		5,
		ROUND_RNE);
  	}));
  } else if (!mode.compare("FLOAT8_STOCHASTIC")) {
 	  //std::cout <<"FLOAT8 Stochastic called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuFloat8StochasticKernel", ([&] {
    		QuantEmuFloat8Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
        	8,
		5,
		ROUND_STOCHASTIC);
  	}));
  } else if (!mode.compare("FLOAT8FDN_RNE")) {
 	  //std::cout <<"FLOAT8 RNE called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuFloat8RNEKernel", ([&] {
    		QuantEmuFloat8Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
        	8,
		5,
		ROUND_RNE);
  	}));
  } else if (!mode.compare("FLOAT8_STOCHASTIC")) {
 	  //std::cout <<"FLOAT8 Stochastic called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuFloat8StochasticKernel", ([&] {
    		QuantEmuFloat8Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
        	8,
		5,
		ROUND_STOCHASTIC);
  	}));

  } else if (!mode.compare("FLOAT16")) {
   	//std::cout <<"FLOAT16 called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuFloat16Kernel", ([&] {
    		QuantEmuFloat16Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
		0);
  	}));
  } else if (!mode.compare("FLOAT16_WITHOUT_DENORMS")) {
   	//std::cout <<"FLOAT16_WITHOUT_DENORMS called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuFloat16WithoutDenormalsKernel", ([&] {
    		QuantEmuFloat16Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
		1);
  	}));
  } else if (!mode.compare("BFLOAT16_RNE")) {
 	  //std::cout <<"BFLOAT16_RNE called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuBFloat16RNEKernel", ([&] {
    		QuantEmuBFloat16Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
		ROUND_RNE);
  	}));
  }
  else if (!mode.compare("BFLOAT16_RTZ")) {
 	  //std::cout <<"BFLOAT16_RNE called " << std::endl;
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuBFloat16RNEKernel", ([&] {
    		QuantEmuBFloat16Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size,
		ROUND_RNTZ);
  	}));
  } else if (!mode.compare("BFLOAT16_RHAZ")) {
 	  //std::cout <<"BFLOAT16_RNE called " << std::endl;
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuBFloat16RNEKernel", ([&] {
    		QuantEmuBFloat16Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size,
		ROUND_RNAZ);
  	}));
  } else if (!mode.compare("BFLOAT16_STOCHASTIC")) {
 	  //std::cout <<"BFLOAT16_STOCHASTIC called " << std::endl; 
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "QuantEmuBFloat16StochasticKernel", ([&] {
    		QuantEmuBFloat16Kernel<scalar_t><<<blocks, threads>>>(
        	input.data<scalar_t>(),
        	//output.data<scalar_t>(),
  		(inplace) ? input.data<scalar_t>() : output.data<scalar_t>(),
        	size, 
		ROUND_STOCHASTIC);
  	}));
  }

  if (!inplace) return output; 
  else return input;
}