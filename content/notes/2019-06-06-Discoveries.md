
* TF has a addons repo which contains things that are not in the core Tensorflow repo. 

* Tensorflow has a models repo which shows best practices.

* Use  tf.summary.histogram in order to record relevant tensors. This can be later examined within TensorBoard.

* In tf 1.13, you can use tf.train.Saver() in order to save the session as a model checkpoint. 

* Items in the TensorBoard graph session that are not needed can be "removed from main graph".

* Items with the same color in Tensorboard have the same substructure.

* summ = tf.summary.merge_all is going to get every summary in the whole graph.
    * This is exported to disk with writer.add_summary(summ, step_no) 
    
* You can place everything in one group by using .* within tensorboard

* Toggle all runs is going to select everything . 

* Relative time is going to show everything from the same start. 

* Beholder looks at the trate of change of the tensors. It is not just what you are seeing but what is also being computed.
    * If columns are similar, this means that there could be redunancy within the network. 
    
### Tensorflow Tensorboard Viz and Debugger
* There is the suggestion that I need to use something similar to the following to get tensorboard debug hook to work

hook = tf_debug.TensorBoardDebugHook("localhost:6064)

* Dandelion looks at an embedding projector for SmartReply5k. 
* He has it set up to show text (the text in this case is the "label")

* Gender relationship of word2vec can be mapped out
Define axis to be the "differences" between different words

2019-06-07

* One way to examine categorical data is to use embeddings. 

2019-06-10

* The OpenAI way of approaching debugging is that you have to look at EVERY line. You have to determine if any given line even has the possibility
of causing poor results in the model. 

2019-06-11

* The more layers and the more neurons in the model, the more powerful the model, but this in turn means that more data is required to train the model.

2019-06-20
* There is the concept of curriculum learning which can be applied to text generation. 
* repr to get the string representation 

https://github.com/zihangdai/xlnet
https://arxiv.org/pdf/1906.08237.pdf

XLNet - it is difficult to train on GPU with 16GB - This would only how a single sequence of length 512. Because of the memory 
constraints, a large number of GPUs (32-128) would be needed in order to train XLNet. 
* Run classifier would need to be used in order to do the fine-tuning

tf.data.Dataset is designed to work with potentially infinite sequences

glorot_uniform initialization 

The output of any prediction is a probability. That probability has to be sampled in order to create the generative model. 

squeeze removes dimensions of size 1 from the tensor. 

you can even look at the loss over one iteration. Make sure everything is running before running the full model.

google puts the tests right there next to the code for keras

For tf.data.Dataset, you can actually enumerate from the dataset as well as take from it

* You can customize the training of a Keras model using GradientTape

* Covariate shift is the way to measure the change in distributions between train and test


2019-06-21 - 
Organization of NLU topics by representation

2019-06-24

Coocurrence matrics 

* Glove and word2vec take care of the reweighting and the dimensionality reduction
* Word x document is going to be a sparse matrix 
* Word x Discourse -  Switchboard Dialog Act Corpus

* Semantic meaning is being derived from these cooccurence matrix
* coccurenence based on the window or cooccurence with scaling 
* larger flatter windows contain more semantic information
* There could be normalization based on distance. 

* KL divergence wants to divide by Q. So you have to add a small value to Q to make sure that the equation does not fail. 
* KL is probabilistic - It has to be a lot of positive values so that the normalization makes sense 

Representation of natural objects with a handful of features that you can measure

Pointwise Mutual Information - observed / expected in log space

Retrofitting 

Modal Adverbs could give some type of signal 


Moritz - 
Political Polling 
What qualities do you look for  - Richer feedback  - He just used PMI and LSA 

2019-06-24 - after Lunch

What is the approach for long texts?
Parse trees does not work well beyond single sentences
RNN as an autoencoder
Doc2Vec when is going to come up with a word vector as well as a document vector 

Sequence Prediction Metrics - 
* Word error rate (WER)
* BLEU
* Perplexity

!! You can text a model by first getting an example batch from the dataset 

In tf.keras.Model, call is the forward execution of the layers. A tf.keras.model is going to have:
* definition of the layers
* Call - Which is the forward pass of the layers


2019-06-25:

https://nostalgebraist.tumblr.com/post/185326092369/the-transformer-explained

* A fully connected network is great when there is no relationship between the features. This works really well
in regression but it would not work as well when looking at a sequence of data (such as what is in text). 

2019-06-27

https://colab.research.google.com/drive/1wYZ21nyAaBsGwY3WktFCI3IKit5EDeV1#scrollTo=wImj6bBHZJQP
https://github.com/hanxiao/bert-as-service/blob/master/example/example5.py
https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/


2019-07-01

CONCEPTS REVIEWED:
Transformer 
Positional Encoding
Batch Normalization
Tensor2Tensor Diagnostic Tools

RELEVANT LINKS:
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb#scrollTo=_fXvfYVfQr2n
https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/position_encoding.ipynb
http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
http://karpathy.github.io/2019/04/25/recipe/
https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb

COMPLETED:
Positional Encoding understanding
x TF 2.0 Transformer 
O Get understanding of transfer learning tutorial
O Make appropriate changes to BertLayer
x Karpathy - A recipe for training neural networks


* The BERT paper states that feature-based approach (similar to the elmo approach) requires fine tuning all parameters.
* Unlike GPT, BERT is bi-directional. 
* ELMO looks at context sensitive features.

* Transfer learning results in SOTA for Information Extraction.

* Transformer uses stacks of variable sized utterances through the use of self-attention layers. 

* If you have a small dataset, you can cache the dataset to memory in order to get a speedup while reading it
** This is done with dataset.cache()

# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb#scrollTo=_fXvfYVfQr2n
* A positional encoding is going to show how similar the meaning of the words are along with the position in the sentence. 

np.newaxis is going to change the shape of the tensor

* Relative position encoding is a linear function of current position encoding
 
 * As seen in the below, tf.math.equal can be used in order to create a mask:
 
 ```python
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions so that we can add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
```

Batch normalization ( Batch norm) standardizes the inputs. It should make the model faster to train. 
* Batch norm should make larger learning rates possible.
* Batch norm SHOULD NOT BE USED WITH DROPOUT
* layers.BatchNormalization is the way to use this with Keras

Transformer - This is a set of encoders and decoders

2019-07-02

CONCEPTS REVIEWED:
Transformer
GPT-2
Bert and Pals - Adapters 
Bert Squad Classification and changing the Bert Output/head

RELEVANT LINKS
http://jalammar.github.io/illustrated-transformer/
https://blog.floydhub.com/gpt2/
https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_41_2
NAACL 2019 - Transfer Learning Tutorial 


* With respect to the transformer model, "multi-headed" attention expands the model’s ability to focus on different positions.

```python
tf.transpose(x, perm=[0, 2, 1, 3])
```
 * tf.transpose can shift the order of the axis

* Generative word prediction could not use the transformer because the transformer requires part of the sentence in order to be able to work 

* BERT has 340M parameters
* GPT-2 has 1.5 billion parameters

https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_41_2
* Slides suggest to keep the network activations static or frozen in the first approach

* Second method is some type of behavioral probe
* SAT analogies are a way of understanding morphology - This is the transformation of words.

[NAACL Transfer learning tutorial](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_50_189)
* Transfer learning tutorial slide 92 suggests that if pretraining was not used then erasing entire words could help to detect what is important
in a sentiment analysis task 

[NAACL Transfer learning tutorial](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_50_189)
* Slide 96 suggests that the construction of probes could be another way to analzye what is happening with the model

* There is discussion about gradual unfreezing. Is this something that I missed?

* Changing the pre-trained weights is called fine-tuning
* Not changing the pretrained weights = feature-extraction, adapters

[Slide 130 - NAACL Transfer learning tutorial](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_50_189)
* This slide suggests that embeddings can be placed before the backbone

* Regardless of the freezing approach, you are going to end up training all of the layers in the end. 
    * Not sure but this might be dependent on the data that you have for feature extraction.
    
* Learning rate warmup is useds in Transformer (Vaswani NIPS 2017)

[Slide 167 - NAACL Transfer learning tutorial](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_50_189)
* Feature Extraction training of pretrained models is said to be slow compared to fine tuning.

[Slide 184 - NAACL Transfer learning tutorial](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_50_189)

2019-07-03

CONCEPTS REVIEWED:
BERT
Squad
Semi-Supervised Cross-View Training
Cloze-driven Pretraining of Self-attention Networks

RELEVANT LINKS: 
[Natural Language Generation Slides](http://thomwolf.io/data/Meetup_Deep_Learning_Paris_2019_01_30.pdf)
[Semi-Supervised Cross-View Training](https://arxiv.org/pdf/1809.08370.pdf)
[Cloze-driven Pretraining of Self-attention Networks](https://arxiv.org/abs/1903.07785)

* It is possible to initialize as you get a new variable. Code looks like:

```python
  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
```

* Tensorflow Hub code is black box. 
  * You don't have access to the source code. 
  * You can't modify the internals of the model (can't add Adapters)
  
2019-07-08

* There is the idea that if I had the final hidden matrix then I could reshape that and build the classifier on top of that.
* It is key to take a look at run classif

2019-07-09

* Shapes of the Keras inputs are matched against the shapes of the input tensors. 
  * Py_func has the ability to generate unknown shaped tensors which causes the Keras model to fail
  * The fix is that you have to use set_shape for each tensor returned from py_func
  
REVIEW OF PREVIOUS WORK

* The estimator actually has to keep track of training
* At the time, I switched from Estimator to Keras

* The estimator would actually tell you what the outputs would be - but those outputs were raw tensors
* Part of the shift was that Keras is a first class citizen in TF 2.0

* Because everything was based off of the original graph. The signature had to be based off of a lot of placeholders
* I could never get a clear idea of the accuracy of the estimator model

* Completion of the tests was a total mess. It was not in a testing framework and run in an ad hoc manner. 
* The process was 10 minutes - 2 hours. Now I can iterate in around 2 minutes. 

* As opposed to a clean interface (py_func - > inputs) there was a really messy interface that required
grpc server and flask to be inside the same docker container

* All of this was with the original data - it was not even looking at postive, negative, and neutral

** In previous work, there was a 
```python
tf.estimator.export.PredictOutput
```

2019-07-10
CONCEPTS REVIEWED:
BERT
tf.keras.metrics.Accuracy

RELEVANT LINKS: 


* The reference implementation does state of the art and it only has one dense layer with activation

WHY EXACTLY IS BERT using a custom optimizer????

* Keras relies on model.fit, model.evauate, model.predict


2019-07-11
CONCEPTS REVIEWED:
tf.keras.Model.fit

RELEVANT LINKS: 

NOTES:
keras fit is traditionally used for data that fits in memory
fit_generator is typically used for streaming batches of data as a generator

It is rumored that tf.layers has gone away in tensorflow 2.0


2019-07-11
CONCEPTS REVIEWED:
tf.estimator tutorial

RELEVANT LINKS: 
https://stackoverflow.com/questions/48295788/using-a-keras-model-inside-a-tf-estimator
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_keras/abalone.py
https://guillaumegenthial.github.io/introduction-tensorflow-estimator.html


NOTES:
Estimators have no explicit session

* https://guillaumegenthial.github.io/introduction-tensorflow-estimator.html
* When feeding strings into your graph, you need to feed the string in bytes

You can open multiple files simultaneously and process simultaneously with zip. This is shown in the following:

```python
 def generator_fn(words, tags):
     with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
         for line_words, line_tags in zip(f_words, f_tags):
             yield parse_fn(line_words, line_tags)
```

* Not sure how to handle the fact that I am generating MULTIPLE predictions

* https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/keras.ipynb
* In order to define a custom layer, you need to have the following
  * build - Create the weights of the layer
  * call - do the forward pass
  * compute output shape
  
```python
 tf.keras.callbacks.LearningRateScheduler # This changes the learning rates
 ```
 
 GradientTape is used to trace operations in order to compute gradients later
 
 * The calculation of the loss needs to capture the gradient in order for gradient descent to occur. This looks like the following:
 
 ```python
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])
```

* https://www.tensorflow.org/guide/datasets
* You can merge datasets together by zipping them 

 ```python
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```

* In eager execution, performance automatically goes to the GPU.  

