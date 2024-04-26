# Malaria_classifier

## Using the Keras Dataset: Malaria Cell Images Dataset (https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

We are making a Convolutional Neural Network (CNN) model using **TensorFlow's Keras API** and training it on this Malaria cell image dataset

Following the above link we can download the dataset directly, but the problem is the dataset lacks a distinct "test" subfolder so to create one, we are using **mover.py**.

So in order, the steps are:

1. Download the image dataset from the above link.
  
2. Use mover.py to randomly select images from the training subdirectory and move them to the test subdirectory, (I have used 5200 images from each class for testing purposes).

3. Use Image1.ipynb to generate the model, the model will also be less susceptible to overfitting due to Data Augmentation.

The performance is very good, out of 10400 random test cases, it was able to correctly predict 9961 images, accuracy is around 95.8%:

![image](https://github.com/Mahos-H/Malaria_classifier/assets/115897153/5e8bab27-bdc9-4530-9e8f-236b7ab02f87)
<img width="959" alt="image" src="https://github.com/Mahos-H/Malaria_classifier/assets/115897153/42aa4172-aa23-482e-96c8-5d9818f3bcce">
