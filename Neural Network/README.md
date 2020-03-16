**Goal**

To implement a single-hidden-layer neural network with a sigmoid activation function for the hidden layer, and a softmax on the output layer. As for the loss function, it'll use cross entropy loss, with a one-hot
vector representing the correct class.

**Dataset**

The neural network is used to label images of handwritten letters. The dataset is a subset of an Optical Character Recognition (OCR) dataset, including only the letters "a", "e", "g", "i", "l", "n", "o", "r", "t", and "u". There are three datasets drawn from this data: a small dataset with 60 samples
per class (50 for training and 10 for test), a medium dataset with 600 samples per class (500 for training and 100 for test), and a large dataset with 1000 samples per class (900 for training and 100 for test). \
Each dataset consists of two csv files—train and test. Each row contains 129 columns separated by commas. The first column contains the label and columns 2 to 129
represent the pixel values of a 16*8 image in a row major format. Label 0 corresponds to "a", 1 to "e", 2 to "g", 3 to "i", 4 to "l", 5 to "n", 6 to "o", 7 to "r", 8 to "t", and 9 to "u".

**Initialize**

To use a deep network, we must first initialize the weights and biases in the network. Two possible initializations are used: \
&emsp;*RANDOM*: The weights are initialized randomly from a uniform distribution from -0.1 to 0.1. The bias parameters are initialized to zero. (init_flag==1) \
&emsp;*ZERO*: All weights are initialized to 0. (init_flag==2)

**Output**

*Prediction of Label*: Predictions of your model on training data (<train out>) and test data (<test out>). \
*Cross Entropy*: After each Stochastic Gradient Descent (SGD) epoch, report mean cross entropy on the training data crossentropy(train) and test data crossentropy(test). \
*Error*: After the final epoch (i.e. when training has completed fully), report the final training error error(train) and test error error(test).


**Command Line Arguments**

As an example, to implemented a 4-hidden units NN on the small data for 2 epochs using zero initialization and a learning rate of 0.1. \
&emsp; $ python neuralnet.py smallTrain.csv smallTest.csv model1train_out.labels model1test_out.labels model1metrics_out.txt \
&emsp; 2 4 2 0.1
