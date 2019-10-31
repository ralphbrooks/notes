---
title: "AWS Code Approach"
author: "Ralph Brooks"
date: 2019-9-04 T12:25:53-04:00
description: "These are general notes on AWS Code Approach"
type: technical_note
draft: false
---

Local Mode Training -> Full Scale Hosted Training

## Amazon Sagemaker Script mode

### Eager
This does not seem to give 1.14
* It does show deployment and hosted training 
* It does show bayesian optimization
  * Bayesian optimization shows ranges for the hyperparamters

### Keras Glove Embeddings

* This requires a P3 or P2 instance
* This uses tf 1.14 in the code leading up to the Sagemaker Estimator

### Distributed training
* This looks like it use

## Amazon Sagemaker Examples


### advanced_functionality/tensorflow_bring_your_own
* This builds and registers a container
* This shows how to take the container to scale training


!!!!! POSSIBLE STRATEGY

What exactly is going to be the benefit here if I bring my own container?

* It almost seems like you deploy to docker and then you do the training that way???
* It says that you only need one image for training and hosting

This is different in the sense that the training code actually gets deployed to docker first

They definitely seem to be using TF serving in the tf_BYO example. 

TESTING STEPS
0. Run through their example exactly as is
1. Do the deployment using TF 1.14
1.1. Take my example and start to mirror that to their example

2. Start to modify the deployment so that you can use multi-gpu training within the Estimator

In theory, after the training completes, there is an automatic save of the model as the SavedModel.pb format


### advanced_functionality/tensorflow_BYOM_iris

### advanced_functionality pytorch_extending_our_containers
* This only gives an example of what happens when the model is training

### sagemaker_batch_transform/tensorflow_cifar-10_with_inference_script
* This is distributed training with Horovod
* It seems that this includes a requirements.txt file.

### sagemake_batch_transform/working_with_tfrecords

* This talks about the buildout of the inference pipeline

### sagemaker-python-sdk

#### keras_script_mode_pipe_mode_horovod
* shows how to take in train/ validation ( dev ) / and eval datasets all in one line into fit
* metrics definition can be sent over to cloudwatch

#### tensorflow-eager-script-mode
* This shows a basic keras model with a hosted endpoint and automatic tunine


###  sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators

* This uses estimators
* Uses sagemaker.tensorflow.Tensorflow with framework='1.12'

### sagemaker-python-sdk/tensorflow_resnet-cifar10_with_tensorboard
* This uses tensorboard
* This makes use of tf.data

### sagemaker-python-sdk/tensorflow_script_mode_training_and_serving
* This uses estimators as part of a 1.14 framework


### sagemaker_batch_transform/tensorflow_cifar-10_with_inference_script
* This is the distributed training with Horovod
* It uses the 1.13 framework though (which I can't use)

## Sagemaker Keras Examples - https://github.com/xkumiyu/sagemaker-keras-example.git

1) How exactly is the pipfile used?
2) How exactly 

### deploy_model.py

* If you have a model that is pretrained, then this is what is pushing that data to the estimator.

