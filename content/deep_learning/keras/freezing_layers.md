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

This might be done for an Embedding layer in Keras so that you don't forget anything that you 
have already learned in the model.

* The large changes in gradient from other parts of the model could be disruptive to this part.