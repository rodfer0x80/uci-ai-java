package uci_neuralnet;

import java.io.FileNotFoundException;

public class Main {

    public static void main(String[] args) {
        System.out.printf("[*] Starting neural network...\n");
        Utility.startTimer();
        NeuralNet network = new NeuralNet(new int[]{Categoriser.INPUT_LAYER_NODE_AMOUNT, Categoriser.FIRST_HIDDEN_LAYER_NODE_AMOUNT, Categoriser.SECOND_HIDDEN_LAYER_NODE_AMOUNT, Categoriser.OUTPUT_DOMAIN});
        Utility.stopTimer();
        System.out.printf("[*] Initialisation time: %s\n", Utility.getTimerns());
        
        try{
        	// train neural net using preprocessed input
        	Utility.startTimer();
        	Dataset set = Categoriser.parseDataset(Categoriser.TRAINING_FILE_PATH);
            Categoriser.trainModel(network, set , Categoriser.TRAINING_EPOCHS_VALUE, Categoriser.TRAINING_LOOPS_VALUE, Categoriser.TRAINING_BATCH_SIZE);
            Utility.stopTimer();
            System.out.printf("[*] Training time: %s\n", Utility.getTimerns());
            
            // test neural net progress using preprocessed and predetermined input
            Utility.startTimer();
            Dataset testSet = Categoriser.parseDataset(Categoriser.TESTING_FILE_PATH);
            Categoriser.testModel(network, testSet);
            Utility.stopTimer();
            System.out.printf("[*] Testing time: %s\n", Utility.getTimerns());
            
        }catch(FileNotFoundException ex) {
            System.out.printf("[x] Error reading input files: %s\n", ex); 
        }
    }
}
