---
title: "Deep Learning Checklist"
author: "Ralph Brooks"
date: 2019-06-02T12:25:53-04:00
description: "Task List of best practicles for a Deep Learning Model"
type: technical_note
draft: false
---

- Define the problem
- Identify a way to reliably measure success against a goal
- Prepare a validation process that you will use to evaluate models
- Vectorize the data
- Develop a model that beats a trivial common sense baseline
- Refine model architecture
- Get your model to overfit

After overfitting use the following to refine model architecture

- Add regularization (dropout) to the model
- Downsize the model to use lower capacity


Use Tensorboard in order to:
* Use tf.summary.scalar for metrics (such as loss functions and metrics)
* Write out respective summaries of input data 
  *  tf.summary.text in order to write out text examples
  *  Use tf.summary.tensor as a generic catchall for any value
  
* Confirm that the appropriate parts of the graph have been moved to the GPU (TensorBoard - Graphs)