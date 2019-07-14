---
title: "Freezing a layer in Keras"
author: "Ralph Brooks"
date: 2019-06-04T12:25:53-04:00
description: "Freezing a layer in Keras"
type: technical_note
draft: false
---

In situations where there is class imbalance, it is helpful to look not only at accuracy, but to look at 

* Precision
* Recall
* F1 scores


Deep learning approches to get better balancing include:

* Assigning weights when doing the fit
* Using focal_loss for the loss method