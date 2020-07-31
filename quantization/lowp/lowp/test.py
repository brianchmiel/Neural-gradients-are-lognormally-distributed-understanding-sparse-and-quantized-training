import torch
from lowp.functional import truncate_bf16, truncate_fp8

x = torch.randn(10).to(device="cuda", dtype=torch.float)
print('FP32:', x)
bf16 = truncate_bf16(x, False, 1)
print('BF16:', bf16)
t_fp8 = truncate_fp8(x, False, 1)
print('FP8:', t_fp8)

itr = 100000
acc = torch.zeros_like(x).type(torch.float64)

for i in range(0,itr):
	t_fp8 = truncate_fp8(x, False, 1)
	acc += t_fp8.type(torch.float64)

print(acc / itr)

acc = torch.zeros_like(x).type(torch.float64)

for i in range(0,itr):
	t_fp8 = truncate_fp8(x, False, 0)
	acc += t_fp8.type(torch.float64)

print(acc / itr)

# fp8 = fp32_to_fp8(x, 1)

