package uci_neuralnet;

import java.util.concurrent.TimeUnit;

public class Utility {

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

	// create set of size n, generate and fill array with random data from weights
    public static Integer[] randomValues(int smallest, int biggest, int size){
        smallest --;
        
        if(size > (biggest - smallest)){
            return null;
        }
        
        Integer[] values = new Integer[size];
        for(int index = 0; index < size; index++){
            int number = (int) (Math.random() * (biggest - smallest + 1) + smallest);
            while(containsValue(values, number)){
                number = (int)(Math.random() * (biggest - smallest + 1) + smallest);
            }
            values[index] = number;
        }
        return values;
    }

   // check if array contains a specific value
   public static <T extends Comparable<T>> boolean containsValue(T[] array, T value){
       for(int index = 0; index < array.length; index++){
           if(array[index] != null){
               if(value.compareTo(array[index]) == 0 ){
                   return true;
               }
           }
       }
       return false;
   }
   

   // get index of highest value in array
   public static int returnIndexOfHighestValue(double[] input){
       int returnIndex = 0;
       for(int iterationIndex = 1; iterationIndex < input.length; iterationIndex++){
           if(input[iterationIndex] > input[returnIndex]){
               returnIndex = iterationIndex;
           }
       }
       return returnIndex;
   }

    // print neuralnet accuracy
    public static void printFinalResults(int goodResults, int totalInputs){
        System.out.println(goodResults + "/" + totalInputs);
        System.out.println("[*] Accuracy: " + (double)((double)goodResults * 100 / (double)totalInputs) + "% ");
    }
}
