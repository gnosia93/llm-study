
Today, PyTorch supports the following backends for running quantized operators efficiently:
* x86 CPUs with AVX2 support or higher (without AVX2 some operations have inefficient implementations), via x86 optimized by fbgemm and onednn (see the details at RFC)
* ARM CPUs (typically found in mobile/embedded devices), via qnnpack
* (early prototype) support for NVidia GPU via TensorRT through fx2trt (to be open sourced)

## 샘플 코드 ##
아래의 코드는 Colab(X64 환경) 또는 맥북 M1(ARM) 에서 실행할 수 있다. ARM 아키텍처에서 실행하는 경우 Quantization 백엔드 엔진을 qnnpack 로 교체해야 한다. (아래 샘플 참조)
```
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    
    def forward(self, x):
        x = self.fc(x)
        return x

model_fp32 = M()
model_fp32
```
```
M(
  (fc): Linear(in_features=4, out_features=4, bias=True)
)
```
```
torch.backends.quantized.engine = 'qnnpack'   <--- 맥북에서 실행하는 경우에만 실행

model_int8 = quantization.quantize_dynamic(
    model_fp32, 
    {torch.nn.Linear},
    dtype=torch.qint8
)
model_int8
```
```
M(
  (fc): DynamicQuantizedLinear(in_features=4, out_features=4, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)
```
```
input_fp32 = torch.randn(4, 4, 4, 4)
input_fp32 = torch.randn(4, 4, 4, 4)
res = model_int8(input_fp32)
```

## Dynamic Quantization on BERT ##

* https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dynamic_quantization_bert_tutorial.ipynb#scrollTo=9sTUmFJfIgN-


## 레퍼런스 ##

* https://pytorch.org/docs/stable/quantization.html
* https://blog.ggaman.com/1028
* https://github.com/aws-samples/aws-sagemaker-intel-quantization/tree/main
* https://github.com/pytorch/QNNPACK
* https://tutorials.pytorch.kr/intermediate/realtime_rpi.html
