
---
title: "Deep Learning Dictionary"
author: "Ralph Brooks"
date: 2019-07-14T12:25:53-04:00
description: "This is a dictionary of deep learning terminology"
type: technical_note
draft: false
---

**Additive Smoothing** : When calculating the maximum likelihood estimate $\theta_j$, you want to make sure that even unlikely possibilities could be generated in the additive model.
Discussed in Generative Deep Learning - July 2019 - David Foster - Chapter 1


**Activations** : These are tne nonlinearities that rae introduced within Dense layers.

* Sigmoid - This is used for multiclass classification (when an item can belong to more than one class ). It is used when you want the values to be between 0 and 1. 
This is represented as $\frac{1}{1 + e^{-x}}$


Discussed in Generative Deep Learning - July 2019 - David Foster - Chapter 2

**Autoregressive Model** : This is a unidirectional model that attempts to predict data from past input. 


**Catastrophic Forgetting** : A situation where the model forgets the task on which it was originally trained.
Discussed in [NAACL 2019 Transfer Learning Tutorial Slides](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5a24b37ba6_3_865)



**Naive Bayes** : This modeling technique makes the assumption that each feature is independent of every other feature. 


**Sequential Adaptation** : Intermediate fine-tuning on related datasets and tasks. 
Discussed in Generative Deep Learning - July 2019 - David Foster - Chapter 1

