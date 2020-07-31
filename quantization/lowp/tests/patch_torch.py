from torchvision.models import resnet50
import torch
import lowp

x = torch.randn(5, device='cuda', requires_grad=True)


def test_functions(x):
    return {'pow': x ** 2.,
            'mul': x * 3,
            'div': x/2,
            'sqrt': x.sqrt(),
            'rsqrt': x.rsqrt(),
            'div_fun': torch.div(x, 2)}


y = test_functions(x)

lowp.enable()
qy = test_functions(x)
lowp.disable()

diff = {}
for k in y.keys():
    diff[k] = float((y[k] - qy[k]).norm())

print(diff)


x = torch.randn(1, 3, 224, 224, device='cuda')

m = resnet50().cuda()


with lowp.Lowp('BF16', warn_patched=True, warn_not_patched=False):
    qy = m(x)

# with lowp.Lowp('FP8'):
#     qy8 = m(x)
# y = m(x)
