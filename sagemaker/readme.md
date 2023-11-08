## 학습 ##

* [Amazon SageMaker 모델 학습 방법 소개](https://www.youtube.com/watch?v=oQ7glJfD-BQ)
* [Deep Learning 모델의 효과적인 분산 트레이닝과 모델 최적화 방법](https://www.youtube.com/watch?v=UFCY8YpyRkI)

#### 분산학습 ####

* Data Parallel
  * [Distributed ML training with PyTorch and Amazon SageMaker - Data Parallel](https://www.youtube.com/watch?v=D9n_GPfcFhc)
    * https://github.com/shashankprasanna/pytorch-sagemaker-distributed-workshop/tree/main

* Model Parallel
  
  [Tensorflow]
  * [Introducing SageMaker Model Parallelism - Model Parallel](https://www.youtube.com/watch?v=eo2zgncnf-M)
  * [Train billion-parameter models with model parallelism on Amazon SageMaker](https://www.youtube.com/watch?v=vv52RsBM8o4)
    * https://github.com/aws/amazon-sagemaker-examples/blob/main/training/distributed_training/tensorflow/model_parallel/mnist/tensorflow_smmodelparallel_mnist.ipynb

  [Pytorch]
  * [At-scale Training with pyTorch and Amazon SageMaker | Andrea Olgiati](https://www.youtube.com/watch?v=ZCbfyPPdmS4)
    * https://github.com/aws/amazon-sagemaker-examples/tree/main/training/distributed_training/pytorch/model_parallel

  [Huggingface Sagemaker SDK]
  * https://github.com/huggingface/notebooks/blob/main/sagemaker/04_distributed_training_model_parallelism/sagemaker-notebook.ipynb

* DeepSpeed
  * https://github.com/aws-samples/training-llm-on-sagemaker-for-multiple-nodes-with-deepspeed
  
    
#### Docs ####
* [sagemaker model parallism library](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html)
* https://github.com/huggingface/notebooks/tree/main/sagemaker
