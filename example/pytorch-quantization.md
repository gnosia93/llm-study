
Today, PyTorch supports the following backends for running quantized operators efficiently:
* x86 CPUs with AVX2 support or higher (without AVX2 some operations have inefficient implementations), via x86 optimized by fbgemm and onednn (see the details at RFC)
* ARM CPUs (typically found in mobile/embedded devices), via qnnpack
* (early prototype) support for NVidia GPU via TensorRT through fx2trt (to be open sourced)



## 레퍼런스 ##

* https://pytorch.org/docs/stable/quantization.html
* https://blog.ggaman.com/1028
* https://github.com/aws-samples/aws-sagemaker-intel-quantization/tree/main
