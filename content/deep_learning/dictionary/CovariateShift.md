
---
title: "Covariate Shift"
author: "Ralph Brooks"
date: 2019-07-28T11:18:53-04:00
description: "Covariate Shift"
type: technical_note
draft: false
---

These are some additional notes that I am taking on the incredible book by David Foster on <a target="_blank" href="https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947?&_encoding=UTF8&tag=ralphbrooks-20&linkCode=ur2&linkId=56166717a12536849e1d3edbbb76330b&camp=1789&creative=9325">Generative Deep Learning</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=ralphbrooks-20&l=ur2&o=1" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> 

This is the change in the distributions of the internal nodes of a deep network. Foster states in chapter 2 of the Generative Deep Learning book that the longer that a network trains, the greater
the possibility that the weights could move from the random initial values and this in turn could lead to <i>NaN</i> errors. 

[Machine Learning Mastery](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/) called this internal covariate shift.




