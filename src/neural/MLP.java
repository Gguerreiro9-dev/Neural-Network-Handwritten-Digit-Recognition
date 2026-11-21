package neural;

import math.Matrix;
import neural.activation.IDifferentiableFunction;

import java.util.Random;

public class MLP {

    private Matrix[] w;      // weights
    private Matrix[] b;      // biases
    private Matrix[] z;      // pre-activation values
    private Matrix[] a;      // activations (yp[])
    private IDifferentiableFunction[] act;
    private int L;           // number of layers

    public MLP(int[] layerSizes, IDifferentiableFunction[] act, int seed) {

        if (seed < 0) seed = (int) System.currentTimeMillis();

        this.L = layerSizes.length;
        this.act = act;

        w = new Matrix[L - 1];
        b = new Matrix[L - 1];
        z = new Matrix[L];
        a = new Matrix[L];

        Random rnd = new Random(seed);

        for (int l = 0; l < L - 1; l++) {

            int nIn = layerSizes[l];
            int nOut = layerSizes[l + 1];

            // He Uniform
            double limit = Math.sqrt(6.0 / nIn);

            double[][] wData = new double[nIn][nOut];
            for (int i = 0; i < nIn; i++)
                for (int j = 0; j < nOut; j++)
                    wData[i][j] = rnd.nextDouble() * 2 * limit - limit;

            w[l] = new Matrix(wData);
            b[l] = new Matrix(new double[1][nOut]);
        }
    }

    // ============================
    // FEED FORWARD
    // ============================
    public Matrix predict(Matrix X) {
        a[0] = X;

        for (int l = 0; l < L - 1; l++) {
            z[l + 1] = a[l].dot(w[l]).addRowVector(b[l]); // pre-activation
            a[l + 1] = z[l + 1].apply(act[l].fnc());       // activation
        }

        return a[L - 1];
    }

    // ============================
    // BACKPROPAGATION CORRETO
    // ============================
    public Matrix backPropagation(Matrix X, Matrix y, double lr) {

        Matrix output = predict(X);

        int m = X.rows();
        double scaledLR = lr / m;

        Matrix[] delta = new Matrix[L];

        // -------- OUTPUT LAYER --------
        Matrix e = y.sub(a[L - 1]);
        delta[L - 1] = e.mult(a[L - 1].apply(act[L - 2].derivative()));

        // -------- HIDDEN LAYERS -------
        for (int l = L - 2; l > 0; l--) {

            Matrix wT = w[l].transpose();
            Matrix temp = delta[l + 1].dot(wT);

            // derivative must use PRE-activation (z), not a[]
            Matrix dAct = z[l].apply(act[l - 1].derivative());

            delta[l] = temp.mult(dAct);
        }

        // -------- WEIGHTS UPDATE -------
        for (int l = 0; l < L - 1; l++) {

            Matrix gradW = a[l].transpose().dot(delta[l + 1]).mult(scaledLR);
            Matrix gradB = delta[l + 1].sumColumns().mult(scaledLR);

            w[l] = w[l].add(gradW);
            b[l] = b[l].addRowVector(gradB);
        }

        return e;
    }

    public double[] train(Matrix X, Matrix Y, double lr, int epochs) {
        double[] mse = new double[epochs];

        for (int e = 0; e < epochs; e++) {
            Matrix err = backPropagation(X, Y, lr);
            mse[e] = err.transpose().dot(err).get(0,0) / X.rows();
        }
        return mse;
    }

    public Matrix getWeights(int layer) { return w[layer]; }
    public Matrix getBias(int layer) { return b[layer]; }

    public void setWeights(int layer, Matrix W) { w[layer] = W; }
    public void setBias(int layer, Matrix B) { b[layer] = B; }
}