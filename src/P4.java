import math.Matrix;
import neural.MLP;
import neural.activation.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class P4 {

    private static double[][] W0, B0, W1, B1;

    // ==============================================================
    // LOAD MATRIX
    // ==============================================================
    private static double[][] loadMatrix(String filename) throws Exception {
        List<double[]> rows = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;

        while ((line = br.readLine()) != null) {
            line = line.replace("{", "").replace("}", "").trim();
            if (line.isEmpty()) continue;

            String[] parts = line.split(",");
            double[] row = new double[parts.length];

            for (int i = 0; i < parts.length; i++)
                row[i] = Double.parseDouble(parts[i].trim());

            rows.add(row);
        }
        br.close();
        return rows.toArray(new double[0][]);
    }

    // ==============================================================
    // MAIN
    // ==============================================================
    public static void main(String[] args) throws Exception {

        // -------------------------------------------------------------
        // 1) CARREGAR PESOS
        // -------------------------------------------------------------
        W0 = loadMatrix("src/W0.txt");
        B0 = loadMatrix("src/B0.txt");
        W1 = loadMatrix("src/W1.txt");
        B1 = loadMatrix("src/B1.txt");

        // -------------------------------------------------------------
        // 2) LER INPUT (20x20 = 400)
        // -------------------------------------------------------------
        Scanner sc = new Scanner(System.in);
        List<double[]> rows = new ArrayList<>();

        while (sc.hasNextLine()) {
            String line = sc.nextLine().trim();
            if (line.isEmpty()) continue;

            line = line.replaceAll(",+", ",");
            String[] parts = line.split(",");
            if (parts.length < 400) continue;

            double[] img20 = new double[400];
            for (int i = 0; i < 400; i++)
                img20[i] = Double.parseDouble(parts[i].trim());

            rows.add(img20);
        }
        sc.close();

        if (rows.isEmpty()) return;

        // -------------------------------------------------------------
        // 3) POOLING 20x20 -> 10x10 -> EXPANDIR PARA 20x20
        // -------------------------------------------------------------
        double[][] X = new double[rows.size()][400];

        for (int i = 0; i < rows.size(); i++) {
            double[] pooled10 = downsample20to10(rows.get(i));
            X[i] = upsample10to20(pooled10);
        }

        // -------------------------------------------------------------
        // 4) MIN–MAX NORMALIZATION
        // -------------------------------------------------------------
        minMaxNormalize(X);

        Matrix evX = new Matrix(X);

        // -------------------------------------------------------------
        // 5) RECRIAR A REDE (400 -> 1)
        // -------------------------------------------------------------
        int[] topology = {400, 32, 1};

        MLP mlp = new MLP(
                topology,
                new IDifferentiableFunction[]{
                        new LeakyReLU(),
                        new Sigmoid() },
                2
        );

        // -------------------------------------------------------------
        // 6) APLICAR PESOS
        // -------------------------------------------------------------
        mlp.setWeights(0, new Matrix(W0));
        mlp.setBias(0, new Matrix(B0));
        mlp.setWeights(1, new Matrix(W1));
        mlp.setBias(1, new Matrix(B1));

        // -------------------------------------------------------------
        // 7) PREDIÇÃO
        // -------------------------------------------------------------
        Matrix pred = mlp.predict(evX);
        pred = pred.apply(new Step().fnc());

        // -------------------------------------------------------------
        // 8) OUTPUT FINAL
        // -------------------------------------------------------------
        for (int i = 0; i < pred.rows(); i++) {
            int cls = (int) pred.get(i, 0);
            System.out.println(cls == 0 ? 2 : 3);
        }
    }

    // ==============================================================
    // 20x20 -> 10x10 (pooling média 2x2)
    // ==============================================================
    private static double[] downsample20to10(double[] img20) {

        double[] img10 = new double[100];

        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {

                int by = y * 2;
                int bx = x * 2;

                double sum = 0.0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                        sum += img20[(by + dy) * 20 + (bx + dx)];

                img10[y * 10 + x] = sum / 4.0;
            }
        }
        return img10;
    }

    // ==============================================================
    // 10x10 -> 20x20 (replicação simples)
    // ==============================================================
    private static double[] upsample10to20(double[] img10) {

        double[] img20 = new double[400];

        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {

                double v = img10[y * 10 + x];

                int by = y * 2;
                int bx = x * 2;

                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                        img20[(by + dy) * 20 + (bx + dx)] = v;
            }
        }
        return img20;
    }

    // ==============================================================
    // MIN–MAX NORMALIZATION
    // ==============================================================
    private static void minMaxNormalize(double[][] X) {

        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        for (double[] row : X)
            for (double v : row) {
                if (v < min) min = v;
                if (v > max) max = v;
            }

        double range = max - min;
        if (range == 0) range = 1.0;

        for (int i = 0; i < X.length; i++)
            for (int j = 0; j < X[i].length; j++)
                X[i][j] = (X[i][j] - min) / range;
    }
}
