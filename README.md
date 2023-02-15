# uci-ai-java
> neural net for uci digit categorisation

### 🏠 [Homepage](/Digits-Recognition-using-Neural-Network/src/RecognizingHandwrittenDigits/RecognizingHandwrittenDigits.java)

## Prerequisite

### Java Development Kit (JDK) 
> JDK version 11 is used for this project as it includes the JavaFX library. Download it [here](https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html).
> 
> For Windows:
```sh
set JAVA_HOME="C:\[Path to folder]\Java\jdk-11.0.14
```
> Enter the Environment Variables in System Properties.
> 
> Add **%JAVA_HOME%\bin** into Path.
```sh
%JAVA_HOME%\bin
```
<p align="center">
  <img height="400" src="https://github.com/rohan-bhautoo/Point-Of-Sales-System/blob/master/Screenshots/Env%20Variable.png">
</p>

## Neural Network
> A neural network-based classifier, called Multi-Layer perceptron (MLP), was used in the project to classify handwritten digits. The MLP consists of three layers which are the input layer, hidden layer and output layer.
<p align="center">
  <img height="300" src="https://github.com/rohan-bhautoo/Digits-Recognition-using-Neural-Network/blob/main/imgs/multi-layer-perceptron.jpg">
</p>

> Each of these layers contain a certain number of nodes which are also called neurons and each node in a layer is connected to all other nodes to the next layer. This can also be referred to as the feed forward network. The number of nodes in the input layer depends upon the number of attributes present in the dataset. For this project, the input attributes are integers in the range ```0..16``` and the last attribute is the class code ```0..9```. The number of nodes in the output layer relies on the number of apparent classes that exist in the dataset. The number of hidden layers is hard to determine as the numbers are selected experimentally. Thereby, the following numbers are given to the first and second hidden layers which can be found in the [RecognizingHandwrittenDigits.java](/Digits-Recognition-using-Neural-Network/src/RecognizingHandwrittenDigits/RecognizingHandwrittenDigits.java) class.

```java
final static int FIRST_HIDDEN_LAYER_NODE_AMOUNT = 26;
final static int SECOND_HIDDEN_LAYER_NODE_AMOUNT = 15;
```
> In MLP, the connection between two nodes consists of a weight. During training, process, it learns the accurate weight adjustment which corresponds to each connection. For the learning purpose, it uses a supervised learning technique named as Backpropagation algorithm.

### Hidden Layers
> Hidden layers of a neural network are just a way to add more neurons in between the input and output layers.
<p align="center">
  <img height="300" src="https://github.com/rohan-bhautoo/Digits-Recognition-using-Neural-Network/blob/main/imgs/single-layer-perceptron.jpg">
</p>

> Different mathematical calculations are implemented to perform the algorithm. First is the dot product function, 
> 
> ![Dot Product Function](https://user-images.githubusercontent.com/47154593/154147044-9a5707a2-a7eb-4b6c-86b1-0a7466cb17bc.png)
> 
> Each of the m features in the input layer is multiplied with a weight (w1, w2, …, wm) and added all together. Then, the hidden output from the hidden neurons (h1, h2, …, hn) are used as input data that has n features. The dot product is finally performed with 1 set of n weights to get the final output y, in the output layer.


### Sigmoid Function
> The Sigmoid function is also called an activation function, which becomes active to allow input signal Z to pass through the neuron if the input is big enough but it limits the output if the input Z is too small.
<p align="center">
  <img height="300" src="https://github.com/rohan-bhautoo/Digits-Recognition-using-Neural-Network/blob/main/imgs/sigmoid-function-curve.png">
</p>

> The graph above shows when the output Z is larger or equal to 0.5, it outputs 1 else if it is smaller than 0.5, it outputs 0. 
> 
> The sigmoid function introduces non-linearity into the neural network model which means that the output from the neuron, which is the dot product of inputs x and weights w plus bias and then put into a sigmoid function, cannot be represented by a linear combination of the input x. This non-linear function produces a new representation of the original data.

### Single-Layer Perceptron vs Perceptron with Hidden Layer
> Perceptron with hidden layers were most preferable to implement because a research paper has shown that single-layer perceptron was unable to learn an XOR gate. For example, it cannot understand the logic that the weather has to be either hot or cold, but not both. Logic gates such as AND, OR, NOT, XOR are very important building blocks of any digital system. However, the multi-layer perceptron was able to learn the XOR gate with backpropagation. Hidden layers can twist the problem in a way that makes it easy for the neural network to classify the problem or pattern. This is very useful for the recognition of handwritten digits.
<p align="center">
  <img height="300" src="https://github.com/rohan-bhautoo/Digits-Recognition-using-Neural-Network/blob/main/imgs/single-layer-perceptron-with-hidden-layer.jpg">
</p>

## Using MLP for Recognition of Handwritten Digits
> The dataset, from the Optical Recognition of Handwritten Digits website, contains 32x32 bitmaps and each of these represent an image of 8x8 pixels. The neural network was implemented in the [NetworkBase.java](/Digits-Recognition-using-Neural-Network/src/RecognizingHandwrittenDigits/NetworkBase.java) class containing 64 nodes in the input layer, 26 in the first hidden layer, 15 in the second hidden layer and 10 in the output layer. The 10 output neurons return in an array where each will be classified a digit from 0 to 9. For example, if the handwritten digit is 0, then the output array will be [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]. If the handwritten digit is 5, the array will be [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]. The neural networks become better by repetitively training them on data and based on the weights in each layer.
> 
> After training the neural network with data from the [first CSV file](/Digits-Recognition-using-Neural-Network/src/RecognizingHandwrittenDigits/Data/cw2DataSet1.csv), the first 4 hidden neurons will be able to recognize the patterns. Then, the [testing CSV file](/Digits-Recognition-using-Neural-Network/src/RecognizingHandwrittenDigits/Data/cw2DataSet2.csv) is fed to the neural network to trigger the hidden neurons and output the correct digit.
<p align="center">
  <img height="300" src="https://github.com/rohan-bhautoo/Digits-Recognition-using-Neural-Network/blob/main/imgs/Trained-neural-network.jpg">
</p>

## Usage
> Compile all the java files using:
```sh
javac *.java
```
> Then, execute the algorithm
```sh
java RecognizingHandwrittenDigits
```

## Author

👤 **Rohan Bhautoo**
* Github: [@rohan-bhautoo](https://github.com/rohan-bhautoo)
* LinkedIn: [@rohan-bhautoo](https://linkedin.com/in/rohan-bhautoo)

## Show your support

Give a ⭐️ if this project helped you!
