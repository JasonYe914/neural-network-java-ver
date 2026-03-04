import java.util.ArrayList;
import java.util.List;

public class Main {
    //Declare and initalize the weights and bias matrix
    private static double[][] W1 = new double[784][64];
    private static double[][] W2 = new double[64][1];
    private static double[][] B1 = new double[1][64];
    private static double[][] B2 = new double[1][1];
    private static int epochs = 1000;
    private static double lr = 0.01;

    public static void main(String[] args) {
        /**
         * Purpose: fill in the weights and bias matrix
         * Preconditions: W1 size of 784 by 64, W2 size of 64, 1
         * Postconditions: Matrix of random numbers and 0's for Weights and bias respectively
         *
         */
        for(int i = 0; i < 784; i++){
            for(int j = 0; j < 64; j++){
                W1[i][j] = Math.random();
            }

        }
        for(int i = 0; i < 64; i++){
            for(int j = 0; j < 1; j++){
                W2[i][j] = Math.random();
            }
        }

        for(int i = 0; i < 1; i++){
            for(int j = 0; j < 64; j++){
                B1[i][j] = 0;
            }
        }

        for(int i = 0; i < 1; i++){
            for(int j = 0; j < 1; j++){
                B2[i][j] = 0;
            }
        }

        int epochs = 1000;
        double lr = 0.01;

        for(int i = 0; i < epochs; i++){

        }
    }

    /**
     * Non-linear activation fucntion for a neural network
     * Returns a number between 0 and 1.
     *
     * @param z a double[][]
     * @return the z or a smaller number of z
     */
    public static double[][] zigmoid(double[][] z){
        for(int i = 0; i < z.length; i++){
            for(int j = 0; j < z[0].length; j++){
                z[i][j] = 1/(1 + Math.exp(-z[i][j]));
            }
        }
        return z;
    }

    /**
     * A relu function is a non-linear activation function
     *
     * @param z 2D array
     * @return a 2D array z with modified changes (if applicable)
     */
    public static double[][] relu(double[][] z){
        for(int i = 0; i < z.length; i++){
            for(int j = 0; j < z[0].length; j++){
                if(z[i][j] < 0) {
                    z[i][j] *= 0.01;
                }
            }
        }
        return z;
    }

    /**
     * Relu derivative function
     *
     * @param z a 2D array
     * @return a 2D array
     */
    public static double[][] relu_derivative(double[][] z){
        for(int i = 0; i < z.length; i++) {
            for (int j = 0; j < z[0].length; j++) {
                if (z[i][j] > 0) {
                    z[i][j] = 1;
                }
                z[i][j] = 0.01;
            }
        }
        return z;
    }

    /**
     * The foward pass of the neural network
     * Weights and bias added
     *
     * @param x a 1D array
     * @return a List of 2D arrays
     */
    public static List<double[][]> forward(int[] x){
        double[][] z1 = new double[784][64];
        for(int i = 0; i < 784; i++){
            for(int j = 0; j < 64; j++){
                z1[i][j] = x[i] * W1[i][j] + B1[i][j];
            }
        }
        double[][] a1 = relu(z1);
        double[][] z2 = new double[64][1];
        for(int i = 0; i < 64; i++){
            for(int j = 0; j < 1; j++){
                z2[i][j] = a1[i][j] * W2[i][j] + B2[i][j];
            }
        }
        double[][] a2 = zigmoid(z2);

        ArrayList<double[][]> var = new ArrayList<>();
        var.add(z1);
        var.add(a1);
        var.add(z2);
        var.add(a2);
        return var;
    }

    /**
     * Loss function to account for accuracy
     * Used for training the neural network in improving accuracy
     *
     * @param y a 1D array
     * @param y_hat a double
     * @return the loss for each iteration in the neural network
     */
    public static double loss(double[] y, double y_hat){
        double eps = 1*Math.pow(10, -8);
        for(int i = 0; i < y.length; i++){
            y[i] = y[i] * Math.log(y_hat + eps) + (1-y[i]) * Math.log(1-y_hat + eps);
        }
        double total_sum = 0;
        for(int i = 0; i < y.length; i++){
            total_sum += y[i];
        }
        double mean = total_sum/(y.length);
        return mean;
    }

    /**
     * Backward propagation of the neural network
     * Trains the AI into making a better prediction
     *
     * TODO: Make sure to pass in 0.1 for lr variable
     * TODO: Implement a function to transpose a matrix
     * @param x
     * @param y
     * @param z1
     * @param a1
     * @param a2
     * @param lr
     */
    public static void backward(double[] x, double[] y, double[][] z1, double[][] a1, double[][] a2, double lr){
        double[][] dz2 = new double[a2.length][a2[0].length];
        double[][] dw2 = new double[W2.length][W2[0].length];

        for(int i = 0; i < dz2.length; i++){
            for(int j = 0; j < dz2[0].length; j++){
                dz2[i][j] = a2[i][j] - y[i];
            }
        }

        //Not sure if this works
        ArrayList<double[]> transpose = new ArrayList<>();
        for(int i = 0; i < a1.length; i++){
            for(int j = 0; j < a1[0].length; j++){
                double temp = a1[i][j];
                a1[i][j] = a1[j][i];
                a1[j][i] = temp;
            }
        }

        double db2;
        double sum = 0;
        for(int i = 0; i < dz2.length; i++){
            for(int j = 0; j < dz2[0].length; j++){
                sum += dz2[i][j];
            }
        }
        db2 = sum/(dz2.length*dz2[0].length);




        for(int i = 0; i < W2.length; i++){
            for(int j = 0; j < W2[0].length; j++){
                W2[i][j] -= lr * dw2[i][j];
            }
        }
        for(int i = 0; i < B2.length; i++){
            for(int j = 0; j < B2[0].length; j++){
                B2[i][j] -= lr * db2;
            }
        }
        for(int i = 0; i < W1.length; i++){
            for(int j = 0; j < W1[0].length; j++){
                W1[i][j] -= lr * dW1[i][j];
            }
        }
        for(int i = 0; i < B1.length; i++){
            for(int j = 0; j < B1[0].length; j++){
                B1[i][j] -= lr * db1;
            }
        }




    }

    /**
     * TODO: Add the prediction method
     */




}