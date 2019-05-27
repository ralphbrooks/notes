---
title: "Using a confusion matrix within Keras"
author: "Ralph Brooks"
date: 2019-05-27T12:25:53-04:00
description: "Using a confusion matrix within Keras"
type: technical_note
draft: false
---

Assuming that y_label are the actual gold standard labels that you want to evaluate, you can use sklearn's confusion matrix
function in order to evaluate a keras model. 



Code would look like the following 

```python
import numpy as np
from sklearn.metrics import confusion_matrix

model = <use your favorite keras model here> 

y_pred = model.predict(test_data, test_labels)
y_pred_max = np.apply_along_axis(lambda x : np.argmax(x) +1, axis =1, arr=y_pred)
cm = confusion_matrix(test_label, y_pred_max)

```