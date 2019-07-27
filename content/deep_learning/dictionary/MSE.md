
---
title: "Mean Squared Error"
author: "Ralph Brooks"
date: 2019-07-27T11:18:53-04:00
description: "Mean Squared Error"
type: technical_note
draft: false
---


These are some additional notes that I am taking on the incredible book by David Foster on <a target="_blank" href="https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947?&_encoding=UTF8&tag=ralphbrooks-20&linkCode=ur2&linkId=56166717a12536849e1d3edbbb76330b&camp=1789&creative=9325">Generative Deep Learning</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=ralphbrooks-20&l=ur2&o=1" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" /> 


MSE stands for Mean Squared Error. The calculation for this is:

$$\Huge MSE = \frac{1}{n}\overset{n}{\underset{i = 1}{\Sigma}} (y_i - p_i)^2$$


This simply shows the difference between the ground truth ($y_i$) and what is predicted ($p_i$). This is typically used when trying to solve regression problems.