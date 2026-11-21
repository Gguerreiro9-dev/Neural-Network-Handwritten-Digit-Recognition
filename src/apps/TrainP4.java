package apps;

import math.Matrix;
import neural.MLP;
import neural.activation.IDifferentiableFunction;
import neural.activation.LeakyReLU;
import neural.activation.Sigmoid;

import java.io.*;
import java.util.*;

public class TrainP4 {

    // ==========================================================
    // MAIN
    // ==========================================================
    public static void main(String[] args) throws Exception {

        // ========= 1. LER DATASET ORIGINAL (20x20 = 400) =========
        double[][] Xraw = readCSV("dataset_last.csv");   // 400 colunas
        double[][] Yraw = readLabels("labels_last.csv");

        int N = Xraw.length;

        // ========= 2. POOLING 2x2 + REPLICAÇÃO (mantém 400) =========
        double[][] X = new double[N][400];
        for (int i = 0; i < N; i++)
            X[i] = poolAndExpand20(Xraw[i]);

        // converter labels 2→0 e 3→1
        double[][] Y = new double[N][1];
        for (int i = 0; i < N; i++)
            Y[i][0] = (Yraw[i][0] == 2 ? 0 : 1);

        // ========= 3. DIVIDIR EM TREINO/TESTE (80/20) =========
        int trainSize = (int) (N * 0.80);

        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < N; i++) idx.add(i);
        Collections.shuffle(idx, new Random(123));

        double[][] Xtrain = new double[trainSize][400];
        double[][] Ytrain = new double[trainSize][1];
        double[][] Xtest  = new double[N - trainSize][400];
        double[][] Ytest  = new double[N - trainSize][1];

        for (int i = 0; i < trainSize; i++) {
            Xtrain[i] = X[idx.get(i)];
            Ytrain[i] = Y[idx.get(i)];
        }
        for (int i = trainSize; i < N; i++) {
            int k = i - trainSize;
            Xtest[k] = X[idx.get(i)];
            Ytest[k] = Y[idx.get(i)];
        }

        // ========= 4. Z-SCORE =========
        ZStats stats = computeZStats(Xtrain);
        applyZScore(Xtrain, stats);
        applyZScore(Xtest, stats);

        Matrix trX = new Matrix(Xtrain);
        Matrix trY = new Matrix(Ytrain);
        Matrix teX = new Matrix(Xtest);
        Matrix teY = new Matrix(Ytest);

        // ========= 5. REDE =========
        int[] topology = {400, 32, 1};

        MLP mlp = new MLP(
                topology,
                new IDifferentiableFunction[]{
                        new LeakyReLU(),
                        new Sigmoid() },
                2
        );

        // ========= 6. PARÂMETROS =========
        int epochs = 30;
        int batchSize = 32;
        double lr = 0.05;

        int patience = 10;
        double minDelta = 1e-2;

        double bestMSE = Double.MAX_VALUE;
        int badCount = 0;

        Matrix[] bestW = new Matrix[2];
        Matrix[] bestB = new Matrix[2];

        // ========= 7. TREINO + EARLY STOPPING =========
        for (int epoch = 0; epoch < epochs; epoch++) {

            List<Integer> batchIdx = new ArrayList<>();
            for (int i = 0; i < trainSize; i++) batchIdx.add(i);
            Collections.shuffle(batchIdx);

            for (int b = 0; b < trainSize; b += batchSize) {

                int end = Math.min(b + batchSize, trainSize);
                int size = end - b;

                double[][] bx = new double[size][400];
                double[][] by = new double[size][1];

                for (int k = b; k < end; k++) {
                    bx[k - b] = Xtrain[batchIdx.get(k)];
                    by[k - b] = Ytrain[batchIdx.get(k)];
                }

                mlp.train(new Matrix(bx), new Matrix(by), lr, 1);
            }

            // -------- TRAIN MSE --------
            Matrix trainPred = mlp.predict(trX);
            Matrix trainErr  = trY.sub(trainPred);
            double trainMSE  = trainErr.transpose().dot(trainErr).get(0, 0) / trX.rows();

            // -------- TEST MSE --------
            Matrix testPred = mlp.predict(teX);
            Matrix testErr  = teY.sub(testPred);
            double testMSE  = testErr.transpose().dot(testErr).get(0, 0) / teX.rows();

            System.out.printf(
                    "Epoch %3d | Train MSE = %.6f | Test MSE = %.6f%n",
                    epoch, trainMSE, testMSE
            );

            if (bestMSE - testMSE > minDelta) {
                bestMSE = testMSE;
                badCount = 0;

                bestW[0] = mlp.getWeights(0);
                bestB[0] = mlp.getBias(0);

                bestW[1] = mlp.getWeights(1);
                bestB[1] = mlp.getBias(1);

            } else {
                badCount++;
            }

            if (badCount >= patience) {
                System.out.println(
                        "\nEARLY STOPPING no epoch " + epoch +
                                " (best test MSE = " + bestMSE + ")"
                );
                break;
            }
        }

        // ========= 8. GUARDAR PESOS =========
        saveMatrix("src/W0.txt", bestW[0]);
        saveMatrix("src/B0.txt", bestB[0]);
        saveMatrix("src/W1.txt", bestW[1]);
        saveMatrix("src/B1.txt", bestB[1]);

        System.out.println("\nPesos guardados!");

        // ========= 9. TESTE FINAL =========
        Matrix finalPred = mlp.predict(teX);

        int correct = 0;
        for (int i = 0; i < finalPred.rows(); i++) {
            int p = finalPred.get(i, 0) < 0.5 ? 0 : 1;
            int r = (int) Ytest[i][0];
            if (p == r) correct++;
        }

        double acc = 100.0 * correct / finalPred.rows();
        System.out.printf("\nACURÁCIA NO TESTE = %.2f%%\n", acc);
    }

    // ==========================================================
    // POOLING 2x2 + REPLICAÇÃO (20x20 -> 20x20)
    // ==========================================================
    private static double[] poolAndExpand20(double[] img20) {

        double[] out = new double[400];

        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {

                int by = y * 2;
                int bx = x * 2;

                double sum = 0.0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                        sum += img20[(by + dy) * 20 + (bx + dx)];

                double avg = sum / 4.0;

                out[(by)     * 20 + (bx)]     = avg;
                out[(by)     * 20 + (bx + 1)] = avg;
                out[(by + 1) * 20 + (bx)]     = avg;
                out[(by + 1) * 20 + (bx + 1)] = avg;
            }
        }
        return out;
    }

    // ==========================================================
    // Z-SCORE
    // ==========================================================
    static class ZStats {
        double[] mean;
        double[] std;
    }

    private static ZStats computeZStats(double[][] X) {
        int d = X[0].length;
        int n = X.length;

        ZStats s = new ZStats();
        s.mean = new double[d];
        s.std = new double[d];

        for (double[] row : X)
            for (int j = 0; j < d; j++)
                s.mean[j] += row[j];

        for (int j = 0; j < d; j++)
            s.mean[j] /= n;

        for (double[] row : X)
            for (int j = 0; j < d; j++) {
                double diff = row[j] - s.mean[j];
                s.std[j] += diff * diff;
            }

        for (int j = 0; j < d; j++) {
            s.std[j] = Math.sqrt(s.std[j] / n);
            if (s.std[j] < 1e-8) s.std[j] = 1.0;
        }

        return s;
    }

    private static void applyZScore(double[][] X, ZStats s) {
        for (int i = 0; i < X.length; i++)
            for (int j = 0; j < X[i].length; j++)
                X[i][j] = (X[i][j] - s.mean[j]) / s.std[j];
    }

    // ==========================================================
    // IO
    // ==========================================================
    public static double[][] readCSV(String filename) throws Exception {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        List<double[]> rows = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            double[] row = new double[parts.length];
            for (int i = 0; i < parts.length; i++)
                row[i] = Double.parseDouble(parts[i]);
            rows.add(row);
        }
        br.close();
        return rows.toArray(new double[0][]);
    }

    public static double[][] readLabels(String filename) throws Exception {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        List<double[]> rows = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null)
            rows.add(new double[]{Double.parseDouble(line.trim())});
        br.close();
        return rows.toArray(new double[0][]);
    }

    public static void saveMatrix(String filename, Matrix m) throws Exception {
        FileWriter fw = new FileWriter(filename);
        for (int i = 0; i < m.rows(); i++) {
            fw.write("{");
            for (int j = 0; j < m.cols(); j++) {
                fw.write(String.format(Locale.US, "%.12f", m.get(i, j)));
                if (j < m.cols() - 1) fw.write(", ");
            }
            fw.write("}\n");
        }
        fw.close();
    }
}