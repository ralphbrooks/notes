---
title: "AWS Approach"
author: "Ralph Brooks"
date: 2019-9-03T12:25:53-04:00
description: "These are general notes on AWS Approach"
type: technical_note
draft: false
---

Blocker: The approach MUST MUST produce a saved model
Blocker: The saved model cannot be in the pb format
Blocker: Must use TF 1.14
Blocker: Must use Python 3
Blocker: Must use Estimator with Tensorflow
Blocker: Must have tensorboard
Blocker: There has to be some way to store the model on S3
Blocker: There has to be some way to use tf.data 

Blocker: There has to be some way to tune the hyperparameters
Blocker: There has to be a way to see the output
Blocker: There has to be some way to use tf.data


### Blocker: Must generate a saved model

Amazon-sagemaker-examples advanced_functionality/tensorflow_BYOM_iris - 
** Just use the regular notebook to create the saved model
** The problem is that you are not coordinating through the SageMaker framework

Amazon-sagemaker-examples sagemaker_batch_transform/tensorflow_cifar-10_with_inference_script
* This is the distributed training with Horovod



### Blocker: Must use TF 1.14

OPTION 1: There is nothing stopping me from running things the same as I ran them before
OPTION 2: Run the training as a docker container
Amazon-sagemaker-examples advanced_functionality/tensorflow_bring_your_own

0) copy the tf-eager script
1) run tf-eager (script mode)
* See if you can get the sagemaker hosted training in that case to run on 1.14
* Is there a way to do this with CPU initially


2) tf-eager - really understand what is going on with the automatic model tuning framework
* Understand if anything is going on with the change in loss function 

3.0) copy tf-sentiment - This seems more likely to show me the setup WITHOUT EAGER EXECUTION
3.1) tf-sentiment - This shows how to have local mode use the local GPU
* The sentiment stuff literally is running off of 1.14
* It almost makes it seem like the process is to do one EPOCH on CPU noteobok
* really investigate the sentiment analysis estimator that is being used

### Blocker: Must Use Python 3

Option 1: Use the docker deployment as seen on amazon-sagemaker-examples advanced_functionality 
tensorflow_bring_your_own


### Blocker: Estimator with Tensorflow
Option1: Amazon-sagemaker-examples advanced_functionality/tensorflow_bring_your_own

### Blocker: Must have tensorboard

amazon-sagemaker-examples /sagemaker-python-sdk/ tesnroflow_resnet_cifar10_with_tensorboard has the tensorboard example


### Blocker: Tf.data

4) It looks like the Horovod stuff has data coming in through the tf.data approach

Low: amazon-sagemaker-examples/sagemaker-python-sdk/tensorflow-pipemode-example sends data to the container through pirpes