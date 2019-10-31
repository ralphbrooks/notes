

### Script mode -long script
* There does not seem to be a way to integrate tensorboard


### Custom Container Deployment

https://aws.amazon.com/blogs/machine-learning/transfer-learning-for-custom-labels-using-a-tensorflow-container-and-bring-your-own-algorithm-in-amazon-sagemaker/

!! This is a viable option though - it would take some work but it would train 
This might work but the deployment to the container - the testing on the container all seems very Labor intenive


### Look at the custom deployment of code

* https://github.com/aws/sagemaker-python-sdk/issues/911

- bash script approach
- we can get to the bash script but we can't execut anything

- Code with the setup approach
1) understand code
2) run this code with the basic GPU
3) start to augment this code




amazon-sagemaker-examples-advanced-functionality-pytorch-extending containers