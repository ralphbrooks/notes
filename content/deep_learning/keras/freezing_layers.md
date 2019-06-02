---
title: "Freezing a layer in Keras"
author: "Ralph Brooks"
date: 2019-05-28T12:25:53-04:00
description: "Freezing a layer in Keras"
type: technical_note
draft: true
---

To freeze a layer in Keras, use:

```python

model.layers[0].trainable = False
```

Notes:

* Typically, the freezing of layers will be done so that weights which are learned in prior stages are not
forgotten in later layers of the model. 
* For example, if you have BERT as one part of a Keras TensorFlow model, that layer might need to be frozen so that 
large changes in gradient that occur during fine tuning do not distrupt the weights that have been learned in BERT. 
