
---
title: "AWS Distributed Training"
author: "Ralph Brooks"
date: 2019-9-03T12:25:53-04:00
description: "These are general notes on AWS Distributed Training"
type: technical_note
draft: false
---


### LOCAL MODE
Source: amazon-sagemaker-script-mode/tf-eagar-sm-scriptmode.ipynb

* Training in local mode requires train_instance_type to be set to 'local'. 
* Use Local if there is a CPU present. use local-gpu if there is a GPU present. 

* This requires the GPU to be running while building
* This is used just to make sure that the code is running correctly. 

```python
model_dir = '/opt/ml/model'
train_instance_type = 'local'
hyperparameters = {'epochs': 5, 'batch_size': 128, 'learning_rate': 0.01}
local_estimator = TensorFlow(entry_point='train.py',
                       source_dir='train_model',
                       model_dir=model_dir,
                       train_instance_type=train_instance_type,
                       train_instance_count=1,
                       hyperparameters=hyperparameters,
                       role=sagemaker.get_execution_role(),
                       base_job_name='tf-eager-scriptmode-bostonhousing',
                       framework_version='1.13',
                       py_version='py3',
                       script_mode=True)
                       
 
```

### LOCAL MODE ENDPOINT
Source: amazon-sagemaker-script-mode/tf-eagar-sm-scriptmode.ipynb

* This is loading a SavedModel or S3 model checkpoint in for testing

### SAGEMAKE HOSTED TRAINING
Source: amazon-sagemaker-script-mode/tf-eagar-sm-scriptmode.ipynb

* training occurs on a separate cluster of machines

### BATCH TRANSFORM FOR LARGE SCALE INFERENCE
Source: Tensorflow sentiment analysis

