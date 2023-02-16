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
        System.out.printf("[*] Starting neural network...\n");
        startTimer();
        NetworkBase network = new NetworkBase(new int[]{Categoriser.INPUT_LAYER_NODE_AMOUNT, Categoriser.FIRST_HIDDEN_LAYER_NODE_AMOUNT, Categoriser.SECOND_HIDDEN_LAYER_NODE_AMOUNT, 10});
        stopTimer();
        System.out.printf("[*] Initialisation time: %s\n", getTimerns());
        
        try{
        	// train neural net using preprocessed input
        	startTimer();
        	Dataset set = Categoriser.parseDataset(Categoriser.TRAINING_FILE_PATH);
            Categoriser.trainModel(network, set , Categoriser.TRAINING_EPOCHS_VALUE, Categoriser.TRAINING_LOOPS_VALUE, Categoriser.TRAINING_BATCH_SIZE);
            stopTimer();
            System.out.printf("[*] Training time: %s\n", getTimerns());
            
            // test neural net progress using preprocessed and predetermined input
            startTimer();
            Dataset testSet = Categoriser.parseDataset(Categoriser.TESTING_FILE_PATH);
            Categoriser.testModel(network, testSet);
            stopTimer();
            System.out.printf("[*] Testing time: %s\n", getTimerns());
            
        }catch(FileNotFoundException ex) {
            System.out.printf("[x] Error reading input files: %s\n", ex); 
        }
    }
}
