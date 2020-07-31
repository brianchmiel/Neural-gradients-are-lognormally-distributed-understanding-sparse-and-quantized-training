#include <torch/extension.h>

// CUDA forward declarations

torch::Tensor trunc_bf16_cuda(
    torch::Tensor input,
    const bool inplace,
    const int roundingMode);


torch::Tensor trunc_fp8_cuda(torch::Tensor input, const bool inplace, const int exp_width, const int man_width, const int exp_bias, const int roundingMode, const int lfsrVal);
torch::Tensor fp32_to_fp8_cuda(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias, const int roundingMode, const int lfsrVal);
torch::Tensor fp8_to_fp32_cuda(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias);
torch::Tensor fp8_to_fp32_cuda(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias);
torch::Tensor quantemu_cuda_forward(torch::Tensor input, std::string mode, bool inplace);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor truncate_bf16(
    torch::Tensor input,
    const bool inplace,
    const int roundingMode) {
  CHECK_INPUT(input);

  return trunc_bf16_cuda(input, inplace, roundingMode);
}

torch::Tensor truncate_fp8(
    torch::Tensor input, const bool inplace, const int exp_width, const int man_width, const int exp_bias, const int roundingMode, const int lfsrVal) {
  CHECK_INPUT(input);
  return trunc_fp8_cuda(input, inplace, exp_width, man_width, exp_bias, roundingMode, lfsrVal);
}

torch::Tensor fp8_to_fp32(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias) {
  CHECK_INPUT(input);
  return fp8_to_fp32_cuda(input, exp_width, man_width, exp_bias);
}


torch::Tensor fp32_to_fp8(torch::Tensor input, const int exp_width, const int man_width, const int exp_bias, const int roundingMode, const int lfsrVal) {
  CHECK_INPUT(input);

  return fp32_to_fp8_cuda(input, exp_width, man_width, exp_bias, roundingMode, lfsrVal);
}

torch::Tensor quantemu(torch::Tensor input, std::string mode, bool inplace) {
  CHECK_INPUT(input);
  return quantemu_cuda_forward(input, mode, inplace);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("truncate_bf16", &truncate_bf16, "BF16 truncation (CUDA)");
  m.def("truncate_fp8", &truncate_fp8, "FP8 truncation (CUDA)");
  m.def("fp8_to_fp32", &fp8_to_fp32, "FP8 to FP32 conversion (CUDA)");
  m.def("fp32_to_fp8", &fp32_to_fp8, "FP32 to FP8 conversion (CUDA)");
  m.def("quantemu", &quantemu, "FP32 to FP* conversion (CUDA)");
}
