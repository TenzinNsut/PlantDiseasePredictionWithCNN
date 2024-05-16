# Plant Disease Prediction with CNN
[Docker] (https://hub.docker.com/r/tenlekshe/plant-disease-prediction/tags)

https://github.com/TenzinNsut/PlantDiseasePredictionWithCNN/assets/105097758/a140ee99-d542-41e6-bb62-153c37cf39a4

This repository contain a Convolutional Neural Network (CNN) model for predicting plant diseases based on input images. The model is trained on the PlantVillage dataset and can classify images into 38 different classes, representing various plant species and their potential diseases.

## Dataset
The [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset is used for training and validation of the plant disease prediction model. It contains a large collection of images of healthy and diseased plant leaves, categorized into different classes based on the plant species and the specific disease.

## Model Architecture
The CNN model architecture consists of the following layers:

1.) Convolutional Layer: Applies 32 filters of size 3x3 with ReLU activation.
2.) Max Pooling Layer: Performs max pooling with a pool size of 2x2 and a stride of 2.
3.) Second Convolutional Layer: Applies 32 filters of size 3x3 with ReLU activation.
4.) Second Max Pooling Layer: Performs max pooling with a pool size of 2x2 and a stride of 2.
5.) Flatten Layer: Converts the output of the convolutional layers into a 1D vector.
6.) Dense Layer: Fully connected layer with 256 units and ReLU activation.
7.) Output Layer: Dense layer with the number of units equal to the number of classes (38) and softmax activation.

## Training
The model is trained using the following configuration:

- Optimizer: Adam
- Loss Function: Categorical Cross-entropy
- Metrics: Accuracy
- Number of Epochs: 5
- Batch Size: 32

The training data is augmented using the ImageDataGenerator from Keras, which applies various transformations such as rescaling, shear, zoom, and horizontal flip to increase the diversity of the training samples.

## Evaluation
The trained model is evaluated on a separate validation set, and the validation accuracy is 88%. The training and validation accuracy and loss curves are plotted to visualize the model's performance during training.
![image](https://github.com/TenzinNsut/PlantDiseasePredictionWithCNN/assets/105097758/ffe4985d-b425-4903-a268-bf621d01aa09)

