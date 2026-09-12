package me.index.ml;

import java.util.Arrays;
import java.util.Random;

public class Net {
    private final int[] sz;
    private final double lr;
    private final int batchSize;
    private final Random rnd;

    private final double[][][] w;
    private final double[][] b;

    private final double[][] a;
    private final double[][] z;
    private final double[][] delta;
    private final double[][][] dw;
    private final double[][] db;
    private int[] perm = new int[0];

    public Net(int[] layerSizes, double learningRate, int batchSize, Random rnd) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("need at least an input and an output layer, got "
                    + Arrays.toString(layerSizes));
        }
        if (layerSizes[0] != 1 || layerSizes[layerSizes.length - 1] != 1) {
            throw new IllegalArgumentException("input and output layers must have size 1, got "
                    + Arrays.toString(layerSizes));
        }
        for (int s : layerSizes) {
            if (s <= 0) {
                throw new IllegalArgumentException("layer sizes must be positive, got "
                        + Arrays.toString(layerSizes));
            }
        }
        if (!(learningRate > 0) || !Double.isFinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be finite and positive, got " + learningRate);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got " + batchSize);
        }

        this.sz = layerSizes.clone();
        this.lr = learningRate;
        this.batchSize = batchSize;
        this.rnd = rnd;

        int layers = sz.length - 1;
        w = new double[layers][][];
        b = new double[layers][];
        dw = new double[layers][][];
        db = new double[layers][];
        z = new double[layers][];
        delta = new double[layers][];
        a = new double[sz.length][];

        a[0] = new double[sz[0]];
        for (int l = 0; l < layers; l++) {
            w[l] = new double[sz[l + 1]][sz[l]];
            for (int i = 0; i < sz[l + 1]; i++)
                for (int j = 0; j < sz[l]; j++)
                    w[l][i][j] = rnd.nextGaussian() * Math.sqrt(2.0 / sz[l]);

            b[l] = new double[sz[l + 1]];
            if (l < layers - 1) {
                Arrays.fill(b[l], 0.3);
            }

            dw[l] = new double[sz[l + 1]][sz[l]];
            db[l] = new double[sz[l + 1]];
            z[l] = new double[sz[l + 1]];
            delta[l] = new double[sz[l + 1]];
            a[l + 1] = new double[sz[l + 1]];
        }
    }

    private void forward(double x) {
        a[0][0] = x;
        int last = sz.length - 2;
        for (int l = 0; l <= last; l++) {
            double[][] wl = w[l];
            double[] al = a[l];
            double[] zl = z[l];
            double[] next = a[l + 1];
            for (int i = 0; i < zl.length; i++) {
                double[] row = wl[i];
                double acc = 0.0;
                for (int j = 0; j < row.length; j++) {
                    acc += row[j] * al[j];
                }
                double v = acc + b[l][i];
                zl[i] = v;
                next[i] = (l < last) ? Math.max(0.0, v) : v;
            }
        }
    }

    public double predict(double x) {
        forward(x);
        return a[sz.length - 1][0];
    }

    public double train(double[] xs, double[] ys, int epochs) {
        if (xs.length != ys.length) {
            throw new IllegalArgumentException("xs and ys must have equal length, got "
                    + xs.length + " and " + ys.length);
        }
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs must not be negative, got " + epochs);
        }
        int n = xs.length;
        if (n == 0 || epochs == 0) return Double.NaN;

        if (perm.length != n) {
            perm = new int[n];
        }
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }

        int layers = sz.length - 1;
        double epochSquaredError = 0.0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = perm.length - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }

            epochSquaredError = 0.0;

            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(start + batchSize, n);

                for (int l = 0; l < layers; l++) {
                    for (double[] row : dw[l]) {
                        Arrays.fill(row, 0.0);
                    }
                    Arrays.fill(db[l], 0.0);
                }

                for (int s = start; s < end; s++) {
                    int idx = perm[s];
                    forward(xs[idx]);

                    double residual = a[sz.length - 1][0] - ys[idx];
                    epochSquaredError += residual * residual;
                    delta[layers - 1][0] = residual;

                    for (int l = layers - 2; l >= 0; l--) {
                        double[] dl = delta[l];
                        double[] dNext = delta[l + 1];
                        double[][] wNext = w[l + 1];
                        double[] zl = z[l];
                        for (int j = 0; j < dl.length; j++) {
                            double acc = 0.0;
                            for (int k = 0; k < dNext.length; k++) {
                                acc += wNext[k][j] * dNext[k];
                            }
                            dl[j] = (zl[j] > 0) ? acc : 0.0;
                        }
                    }

                    for (int l = 0; l < layers; l++) {
                        double[] dl = delta[l];
                        double[] al = a[l];
                        double[][] dwl = dw[l];
                        double[] dbl = db[l];
                        for (int i = 0; i < dl.length; i++) {
                            double di = dl[i];
                            double[] row = dwl[i];
                            for (int j = 0; j < al.length; j++) {
                                row[j] += di * al[j];
                            }
                            dbl[i] += di;
                        }
                    }
                }

                double step = lr / (end - start);
                for (int l = 0; l < layers; l++) {
                    double[][] wl = w[l];
                    double[][] dwl = dw[l];
                    for (int i = 0; i < wl.length; i++) {
                        double[] wRow = wl[i];
                        double[] dRow = dwl[i];
                        for (int j = 0; j < wRow.length; j++) {
                            wRow[j] -= dRow[j] * step;
                        }
                        b[l][i] -= db[l][i] * step;
                    }
                }
            }
        }
        return epochSquaredError / n;
    }
}
