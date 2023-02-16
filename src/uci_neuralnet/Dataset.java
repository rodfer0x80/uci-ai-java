package uci_neuralnet;
import java.util.ArrayList;


// build and parse data from csv files
// and build set in memory
public class Dataset {
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;

    private ArrayList<double[][]> dataset = new ArrayList<>();

    public Dataset(int INPUT_SIZE, int OUTPUT_SIZE) {
        this.INPUT_SIZE = INPUT_SIZE;
        this.OUTPUT_SIZE = OUTPUT_SIZE;
    }

    // add array of parsed data to dataset
    public void addData(double[] in, double[] expected) {
        if(in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE) return;
        dataset.add(new double[][]{in, expected});
    }

    // pulls from dataset in n batches
    public Dataset extractBatch(int size) {
        if(size > 0 && size <= this.size()) {
            Dataset set = new Dataset(INPUT_SIZE, OUTPUT_SIZE);
            Integer[] ids = Utility.randomValues(0,this.size() - 1, size);
            for(Integer i:ids) {
                set.addData(this.getInput(i),this.getOutput(i));
            }
            return set;
        }else return this;
    }

    // get dataset size
    public int size() {
        return dataset.size();
    }

    // pick at choice n from dataset input 
    public double[] getInput(int index) {
        if(index >= 0 && index < size())
            return dataset.get(index)[0];
        else return null;
    }

    // pick at choice n from dataset output 
    public double[] getOutput(int index) {
        if(index >= 0 && index < size())
            return dataset.get(index)[1];
        else return null;
    }

    // get input size
    // deadcode?
    public int getINPUT_SIZE() {
        return INPUT_SIZE;
    }

    // get output size
    // deadcode?
    public int getOUTPUT_SIZE() {
        return OUTPUT_SIZE;
    }
}
