Group Project 4: The Identification Game (by Team Poisson)
=================================================================================

.. contents::

Synopsis
---------

This project is an image classification problem consisting of Natural Images. Our task is to train a classifier to take an image as an input and output a class corresponding to the image. We have focused on using transfer learning on convolutional neural networks (CNNs) to create our model, PoissoNet.
The dataset our model is trained on consists of 100,000 images categorised into 200 categories.


Our two submitted classifiers
-------------------------------
Our following two submissions provided the best F1 scores on the dataset.

Inception-v3  with test time augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Based on tests on a variety of networks suitable for the given dataset, we found Inception-v3 provided the highest F1 score on our validation set. Inception-v3 is based on GoogLeNet (aka Inception-v1). It is a very deep CNN with 42 layers yet fewer parameters than other popular CNNs, making a lower error rate possible. It is also trained on ImageNet, making it suitable for our natural image dataset.
Our submitted model uses transfer learning by loading the pre-trained model, freezing parameters of the first five layers, and finetuning the rest of the layers to the dataset. We also don't use the auxiliary output in our submitted model. Although we experimented with data augmentation using both PyTorch and Albumentations, we decided to use test-time augmentation (TTA) for our predictions.
Test-time augmentation consists of creating augmented copies of a selection of images from the test set and having the model make predictions for these. The final prediction is then an average prediction of Inception-v3 and TTA.


Inception v3 + ResNet18
~~~~~~~~~~~~~~~~~~~~~~~~
One of the pre-trained models we had trained was ResNet18, which also provided a high F1 score. ResNet18 has 18 layers (hence the name) and also trained on ImageNet. This made it a good candidate for training on our dataset. To improve the F1 scores of both Inception-v3 and ResNet18, we decided to average out the two predictions.

Using PoissoNet
-------------------
Loading the dataset
~~~~~~~~~~~~~~~~~~~~~
The dataset is not included in the repository as it is too large to be stored on GitHub. Instead, a user wishing to train or test our models should create a ./train/ or ./test/ directory in PoissoNet.

Choosing the model and the hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Our model trainer allows the user to choose between 4 pre-trained models: ResNet18, ResNet50, GoogleNet, Inception-v3. We found these models to have the highest F1 scores when training and testing on the dataset.
The user can also choose whether they want data augmentation. There are two choices: using PyTorch transforms or Albumentations. The transforms themselves can be adjusted in the code. Note that Inception-v3 has its built-in transforms so no transforms should be done on top of that.
Once a model has been chosen, the user can decide on the following hyperparameters:
* lr: Learning rate. Default value 1e-2.
* momentum: Momentum. Default value 0.4.
* batch_size: Batch size of the training dataset. Default value 64.
* test_batch_size: Test batch size for the validation and test datasets. Default value 100.
* n_epochs: Number of epochs to run the model for. Default value 30.
* weight_decay: Weight decay. Default value 1e-3.

Note that the default values are the values used in our submission, except for the number of epochs. For our submitted models, we had used 3-6 epochs.

Training the model
~~~~~~~~~~~~~~~~~~~~
Once the model and its hyperparameters have been set, simply run the entire Trainer.ipynb notebook. Our trainer splits the training dataset to have 90% training and 10% validation. The notebook will provide a live plot of the accuracy as well as the log loss for both the training and validation to keep track of how the model is doing. Finally, it will save the model for prediction.

Classifying images
~~~~~~~~~~~~~~~~~~~
Once the model has been trained and saved, we can begin classifying the images using Predictor.ipynb. The user can also use one of our trained models in the Models directory. By default the user can choose to either predict with Inception-v3 with TTA or an average prediction of Inception-v3 and Resnet18. The notebook provides the functions to do ensemble prediction of up to 3 models and perform TTA on any model.
