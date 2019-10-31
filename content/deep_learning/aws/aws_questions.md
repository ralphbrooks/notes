
---
title: "AWS Questions"
author: "Ralph Brooks"
date: 2019-9-08 T12:25:53-04:00
description: "Research Questions on AWS"
type: technical_note
draft: false
---

* How does AWS Deal with Batch Size?


=====================

Issue: How does training steps work when you have epoch set in the hyperparmeter?

Issue: What are the implications of using TensorFlow 
[NEW] What does the code look like if I am using script mode with 1.14?
[NEW] What things can't you do if you are using script mode?
* Is it best advised to write the model to S3? Read the tensorboard using that?

Issue: Combining of data
* Is it better to use tf.data or to do the ingestion straight from /opt/model/data?


Issue: How does one deploy models with SageMaker-Python-SDK that relies on external dependencies?

Relevant Questions:
* I need the architect to show me how external dependencies (“pip requirements”) are populated within a standard TensorFlow container or a custom container. 
  * Currently I have a model that relies on the tensorflow_hub library to “tokenize” words prior to model prediction.
  * How is tensorflow_hub added into SageMaker containers?
  * How is the deployment of containers (as Endpoints) impacted by the design decision to use a standard or custom container. 



Issue: I need a walkthrough of the architecture of distributed training within SageMaker-Python-SDK.

Relevant Questions:
* What are the underlying assumptions about the TensorFlow models that use distributed training?
* Does the distributed training act differently if spot is used relative to regular instances?

* Are there any cases where it makes more sense to use Distributed TensorFlow (https://www.tensorflow.org/guide/distribute_strategy) over the code in sagemaker.tensorflow.TensorFlow ?

Issue: If I use just one large script, then how exactly do I use Tensorboard with that?
* [NEW] What are the best practices to get Tensorboard to persist even after the training has completed?
* The URI has to be accessible - how exactly would this work in a restricted environment?

Issue: Importing separate directories
* How do you import other directories into the container?

Issue: How can I get feedback on the structure of my code?
