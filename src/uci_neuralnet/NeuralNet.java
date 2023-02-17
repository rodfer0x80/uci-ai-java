package uci_neuralnet;


public class NeuralNet {

    public final int[] NETWORK_LAYER_SIZE;
    public final int INPUT_LAYER_SIZE;
    public final int OUTPUT_LAYER_SIZE;
    public final int NETWORK_SIZE;
       
    // 1st dimension :: layer index
    // 2nd dimension :: neuron index
    private double[][] output;
    
    // 1st :: layer
    // 2nd :: neuron
    // 3rd :: previous layer neuron (output specific)
    private double[][][] weights;
    
    //1st :: layer
    // 2nd :: neuron
    private double[][]bias;
    
    private double[][] error_signal;
    private double[][] output_derivative;
    
     // build neural net with l1(input), l2, l3 (output) and output domain sizes
    public NeuralNet(int[] NETWORK_LAYER_SIZE){
        this.NETWORK_LAYER_SIZE = NETWORK_LAYER_SIZE;
        this.INPUT_LAYER_SIZE = NETWORK_LAYER_SIZE[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZE.length;
        // 1st layer index is 0
        this.OUTPUT_LAYER_SIZE = NETWORK_LAYER_SIZE[NETWORK_SIZE-1];
        //=output[layer][neuron]
        this.output = new double[NETWORK_SIZE][];
        //=weights[layer][neuron][prevNeuron]
        this.weights = new double[NETWORK_SIZE][][];
        //=bias[layer][neuron]
        this.bias = new double[NETWORK_SIZE][];
        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];
        
        for(int index = 0; index < NETWORK_SIZE; index++){
            this.output[index] = new double[NETWORK_LAYER_SIZE[index]];
            this.error_signal[index] = new double[NETWORK_LAYER_SIZE[index]];
            this.output_derivative[index] = new double[NETWORK_LAYER_SIZE[index]];
            this.bias[index] = buildRandomArray(NETWORK_LAYER_SIZE[index], Categoriser.BIAS_RANGE_SMALLEST, Categoriser.BIAS_RANGE_BIGGEST);
            // exclude the input layer[0] 
            if(index > 0){
                // weights for a layer, specific from previous layer.
                weights[index] = buildRandomArray(NETWORK_LAYER_SIZE[index], NETWORK_LAYER_SIZE[index-1], Categoriser.WEIGHTS_RANGE_SMALLEST, Categoriser.WEIGHTS_RANGE_BIGGEST);
            }
        }
    }
   
    // train model on data for all n iterations
    public void train(Dataset set, int loops, int batch_size){
        if(set.INPUT_SIZE != INPUT_LAYER_SIZE || set.OUTPUT_SIZE != OUTPUT_LAYER_SIZE){
            return;
        }
        for(int index = 0; index < loops; index++){
            Dataset batch = set.extractBatch(batch_size);
            for(int b = 0; b < batch_size; b++){
                this.training(batch.getInput(b), batch.getOutput(b), Categoriser.LEARNING_RATE);
            }
        }
    }

    // train model by getting nn output, caculating error from label result
    // and update weights to improve accuracy on next nn output response
    // (only runtime no mem dump)
    public void training(double[] input, double[] target, double learningRate){
        if(input.length != INPUT_LAYER_SIZE || target.length != OUTPUT_LAYER_SIZE){
            return;
        }
        calculateOutput(input);
        backpropError(target);
        updateWeights(learningRate);
    }
    
    // get response output from nn based on given input
    public double[] calculateOutput(double[] input){
        if(input.length != this.INPUT_LAYER_SIZE){
            return null;
        }
        // index 0 is not a layer but a buffer for input to the layer
        this.output[0] = input;
        for(int layer = 1; layer < NETWORK_SIZE; layer ++){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZE[layer]; neuron++){
                double sum = 0;
                for(int previousNeuron = 0; previousNeuron < NETWORK_LAYER_SIZE[layer-1];
                        previousNeuron++){
                    sum += output[layer-1][previousNeuron] * 
                            weights[layer][neuron][previousNeuron];
                }
                sum += bias[layer][neuron];
                
                output[layer][neuron] = sigmoidFunction(sum);
                output_derivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
                
            }
        }
        return output[NETWORK_SIZE-1];
    }

    // calculate gradient error for weights 
    public void backpropError(double[] target){
        for(int neuron = 0; neuron < NETWORK_LAYER_SIZE[NETWORK_SIZE-1]; neuron++){
            error_signal[NETWORK_SIZE-1][neuron] = (output[NETWORK_SIZE-1][neuron] - target[neuron]) 
                    * output_derivative[NETWORK_SIZE-1][neuron];
        }
        for(int layer = NETWORK_SIZE-2; layer > 0; layer--){
            for(int neuron = 0;  neuron < NETWORK_LAYER_SIZE[layer]; neuron++){
                double sum = 0;
                for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZE[layer+1]; nextNeuron++){
                    sum += weights[layer+1][nextNeuron][neuron] * error_signal[layer+1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivative[layer][neuron];
            }
        }
    }

    // gradient descent learning for weights based on nn calculation and properr
    public void updateWeights(double learningRate){
        for(int layer = 1; layer < NETWORK_SIZE; layer++){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZE[layer]; neuron++){
                
                double delta = - learningRate *  error_signal[layer][neuron];
                bias[layer][neuron] += delta;
                
                for(int previousNeuron = 0; previousNeuron < NETWORK_LAYER_SIZE[layer-1]; previousNeuron++){
                    //weights[layer][neuron][previousNeuron]
                    weights[layer][neuron][previousNeuron] += delta * output[layer-1][previousNeuron];
                }
            }
        }
    }
    
    // avg square diff between model and reference value
    public double MeanSquaredError(double[] input, double[] target){
        if(input.length != INPUT_LAYER_SIZE || target.length != OUTPUT_LAYER_SIZE){
            return 0;
        }
        calculateOutput(input);
        double error = 0;
        for(int index = 0; index < target.length; index++){
            error += (target[index] - output[NETWORK_SIZE-1][index]) * (target[index] - output[NETWORK_SIZE-1][index]);
        }
        return error / (2D * target.length);
    }
    
    // avg square diff between model and reference value
    public double MeanSquaredError(Dataset set){
        double error = 0;
        for(int index = 0; index < set.size(); index++){
            error += MeanSquaredError(set.getInput(index), set.getOutput(index));
        }
        return error / set.size();
    }
    

    // sigmoid function of n
    private double sigmoidFunction(double inputValue){
        return 1D /(1 + Math.exp(-inputValue));
    }
    
	// create array of size n, generate and fill array with random data from weights
    public static double[] buildRandomArray(int range, double smallest, double biggest){
        if(range < 1){
            return null;
        }
        double[] returnArray = new double[range];

        for(int index = 0; index < range; index++){
            returnArray[index] = generateRandomValue(smallest, biggest);
        }
        return returnArray;
    }
    
	// create 2d array of size x by y, generate and fill array with random data from weights
    public static double[][] buildRandomArray(int rangeX, int rangeY, double smallest, double biggest){
        if(rangeX < 1 || rangeY < 1){
            return null;
        }
        double[][] returnArray = new double[rangeX][rangeY];
        for(int index = 0; index < rangeX; index++){
            returnArray[index] = buildRandomArray(rangeY,smallest, biggest);
        }
        return returnArray;
    }

    // get random n
    public static double generateRandomValue(double smallest, double biggest){
        return Math.random()*(biggest - smallest) + smallest;
    }

}
