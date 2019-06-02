---
title: "Deep Dive on Keras Embedding Layers"
author: "Ralph Brooks"
date: 2019-05-28T12:25:53-04:00
description: "Technical notes on the intuition for Keras Embedding Layers"
type: technical_note
draft: true
---

* It seems that each batch has to have the same length.
* Different batches can have different lengths.

Like any other layer, the Keras Embedding layer can have its parameters altered during training.

* Depending on the layer that comes after this. It may be worth Flattening this layer

```python
model.add(Flatten())
```

