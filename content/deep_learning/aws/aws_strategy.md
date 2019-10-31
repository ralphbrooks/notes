    
---
title: "AWS Strategy"
author: "Ralph Brooks"
date: 2019-9-04 T12:25:53-04:00
description: "These are general notes on Strategy"
type: technical_note
draft: false
---

*     use sagemaker-python-sdk / tensorflow_script_mode_quickstart
*     Use the GPU to make sure that training fully completes
*     confirm that the output is being written out to S3
        * Code is written out as a checkpoint


The problem that I see here is that code is getting written out as checkpoints

-2.2    inspect the code

-2.3    Build off of this base example by using tensorboard to inspect

-3. Replicate this when using my code

! Now start to think through the robust training going on with all of this


Use the jumpstart 
Use Amazon-sagemaker-examples advanced_functionality/tensorflow_bring_your_own
Use Amazon-sagemaker-examples advanced_functionality/tensorflow_BYOM_iris


0. Run through their example exactly as is FOR BYO
1. Do the deployment using TF 1.14
1.1. Take my example and start to mirror that to their example

2. Start to modify the deployment so that you can use multi-gpu training within the Estimator

2.05 Expand on BYO with the tensorboard example
* This might be straightforward because the estimator is writing out the .pb file

2.1 Use amazon-sagemaker-examples/sagemaker_batch_transform/tensorflow_cifar-10-with-inference to start
to apply what is going on with Horovod distribution parameters 

!!! At this point you should have enough to be able to extract the saved model

3.05 - Only after this should you start to think about training in parallel. 

3.1 Use transform job for offline inference

4. Think about deployment with elastic inference

