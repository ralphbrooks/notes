
## Installation and requirements 

https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api


## General notes on training

Nick's code actually calls Tensorflow\models\research\object_detection\model_main_tf2.py

There HAS TO be something in model_main_tf2 that can be used - because THIS IS CREATING THE MODEL BEFORE IT IS SAVED

* Maybe there is a way to get the model back to the state it was in at the point where it was saved


--model_dir=Tensorflow\workspace\models\my_ssd_mobnet_tuned --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet_tuned\pipeline.config --num_train_steps=3000

object_detction/model_lib_v2.py contains the training code in train_loop


## Notes on train loop

* processes the config 
* THIS BUILDS THE MODEL AND THE OPTIMIZER




## General Approach to running the model
https://towardsdatascience.com/is-google-tensorflow-object-detection-api-the-easiest-way-to-implement-image-recognition-a8bd1f500ea0

Download the frozen model (.pb — protobuf) and load it into memory

Use the built in helper code to load labels, categories, visualization tools etc.

Open a new session and run the model on an image
