package uci_neuralnet;

import java.io.FileNotFoundException;
import java.util.concurrent.TimeUnit;

public class Main {
	
	public static long t = 0;
	
	public static void startTimer() {
		t = System.nanoTime();
	}
	
	public static void stopTimer() {
		t = System.nanoTime() - t;
	}
	
	public static long getTimers() {
		long ts = TimeUnit.SECONDS.convert(t, TimeUnit.NANOSECONDS);
		return ts;
	}
	
	public static float getTimerns() {
		float tns = TimeUnit.NANOSECONDS.convert(t, TimeUnit.NANOSECONDS);
		return tns;
	}

    public static void main(String[] args) {
    	// model 1 for two fold testing
        System.out.printf("[*] Running model 1...\n");
        NeuralNet model1 = new NeuralNet(new int[]{Categoriser.INPUT_LAYER_NODE_AMOUNT, Categoriser.FIRST_HIDDEN_LAYER_NODE_AMOUNT, Categoriser.SECOND_HIDDEN_LAYER_NODE_AMOUNT, Categoriser.OUTPUT_DOMAIN});        
        try{
        	startTimer();
        	Dataset set1 = Categoriser.parseDataset(Categoriser.DATASET_1_FILE_PATH);
            Categoriser.trainModel(model1, set1 , Categoriser.TRAINING_EPOCHS_VALUE, Categoriser.TRAINING_LOOPS_VALUE, Categoriser.TRAINING_BATCH_SIZE);
            stopTimer();
            System.out.printf("[*] Training: %s\n", getTimerns());
            
            startTimer();
            Dataset testSet1 = Categoriser.parseDataset(Categoriser.DATASET_2_FILE_PATH);
            Categoriser.testModel(model1, testSet1);
            stopTimer();
            System.out.printf("[*] Testing: %s\n", getTimerns());
            
        }catch(FileNotFoundException ex) {
            System.out.printf("[x] Error reading input files: %s\n", ex); 
        }
        
        // model 2
        System.out.printf("[*] Running model 2...\n");
        NeuralNet model2 = new NeuralNet(new int[]{Categoriser.INPUT_LAYER_NODE_AMOUNT, Categoriser.FIRST_HIDDEN_LAYER_NODE_AMOUNT, Categoriser.SECOND_HIDDEN_LAYER_NODE_AMOUNT, Categoriser.OUTPUT_DOMAIN});        
        try{
        	startTimer();
        	Dataset set2 = Categoriser.parseDataset(Categoriser.DATASET_1_FILE_PATH);
            Categoriser.trainModel(model2, set2 , Categoriser.TRAINING_EPOCHS_VALUE, Categoriser.TRAINING_LOOPS_VALUE, Categoriser.TRAINING_BATCH_SIZE);
            stopTimer();
            System.out.printf("[*] Training: %sns\n", getTimerns());
            
            startTimer();
            Dataset testSet2 = Categoriser.parseDataset(Categoriser.DATASET_2_FILE_PATH);
            Categoriser.testModel(model2, testSet2);
            stopTimer();
            System.out.printf("[*] Testing: %sns\n", getTimerns());
            
        }catch(FileNotFoundException ex) {
            System.out.printf("[x] Error reading input files: %sns\n", ex); 
        }
    }
}
