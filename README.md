# Convolutional-Neural-Network
Using __Tensorflow__ to calculate error and confusion matrix of CIFAR-10 Dataset

##__Description__

The purpose of the this assignment is to practice with the convolutional neural networks.
Your task is to Implement a convolutional neural network using Tensorflow to recognize the classes in CIFAR 10 data set. The structure of your neural network, with three convolutional layers, should be as shown below.

Layer 1:

  - **INPUT** : [32x32x3]
  - **CONV** : [32x32xM]
  - **Number of filters**: F1;
  - **kernel size**: K1 x K1
  - **Stride**: 1;
  - **Padding**: "Same"
  - **Activation** : RELU
  - **Max Pool**: Size: 2 x 2

Layer 2:

   - **CONV**:  
   - **Number of filters**: F2 ; 
   - **kernel size**: K2x K2 
   - **Stride**: 1; 
   - **Padding**: "Same"
   - **Activation**: RELU
   - **Max Pool**: Size: 2 x 2

Layer 3:
   - **CONV**:  
   - **Number of filters**: 32 ;  
   - **kernel size**: 3 x 3  
   - **Stride**: 1; 
   - **Padding**: "Same"
   - **Activation**: RELU
   - **Max Pool**: Size: 2 x 2

Final Layer:
   - **Fully connected**


 
### Sliders:
- **"Alpha"**: (Learning rate) Range should be between 0.000 and 1.0. Default value = 0.1 increments=.001
- **"Lambda"**: (Weight regularization). Range should be between 0.0 and 1.0. Default value = 0.01 Increments=0.01
- **"F1"**: Number of filters in the first layer. Range 1 to 64. Default value=32  increment=1
- **"K1"**: Kernel size for the filters in the first layer. Range 3 to 7. Default value=3, increment=2.
- **"F2"**: Number of filters in the second layer. Range 1 to 64. Default value=32  increment=1
- **"K2"**: Kernel size for the filters in the first layer. Range 3 to 7. Default value=3, increment=2.
- **"Training Sample Size (Percentage)"**: This integer slider allows the user to select the percentage of the samples to be used for training. range 0% to 100%. Default value should be 20% which means the 20% of the training samples, will be used for training. Note that this slider does not effect the number of samples for testing. The number of samples for testing is always 10000
### Buttons:
- **"Adjust Weights (Train)"**: When this button is pressed the training should be applied for 1 epoch and the output plot and the confusion matrix should be updated accordingly. Calculations of the error rate should be done after all the samples have been processed. In other words, go through all the current training samples one time and adjust the weights and biases accordingly. Once all the training samples have been processed (one round), freeze the weights and biases, run the test set through the network and update the error rate plot and the confusion matrix .
- **"Reset Weights"**. When this button is pressed all weights should be reset to random numbers and the display should be updated accordingly..
 
 ### Notes: 
- The output of your program should include the plot of the error rate (in percent) and the confusion matrix (in percent) for the test data. The plot and the confusion matrix should be updated after each epoch.
- The CIFAR-10 data set consists of 60000 32×32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. You can download the CIFAR 10 dataset from the following link: cifar-10  
- You may use Keras.
- When your program starts it should automatically read the CIFAR 10 data, randomize the weights, and display the confusion matrix. The CIFAR 10 data should be in a directory called "Data".
- The activation function of the output layer is linear.
- When submitting your assignment submit what is needed for running the program including the source codes, trained weights, etc. but DO NOT submit the CIFAR 10 data set. This means that your submission should include a directory called "Data" but that directory should be empty. At the time of grading your grader will copy the CIFAR 10 data set to the "Data" directory and then run your program.
- Make sure that you follow the submission guidelines to submit your code to Blackboard.



 
