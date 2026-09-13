package me.index.ml;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;

public abstract class Net implements Model {
    private final int[] sz;
    private final double lr;
    private final int batchSize;
    private final Loss loss;
    private final Random rnd;

    private final double[][][] w;
    private final double[][] b;

    private final double[][] a;
    private final double[][] z;
    private final double[][] delta;
    private final double[][][] dw;
    private final double[][] db;
    private int[] perm = new int[0];

    protected Net(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd,
                  Init init, double hiddenBias) {
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
        if (loss == null) {
            throw new IllegalArgumentException("loss must not be null");
        }
        if (rnd == null) {
            throw new IllegalArgumentException("rnd must not be null");
        }
        if (init == null) {
            throw new IllegalArgumentException("init must not be null");
        }
        if (!Double.isFinite(hiddenBias)) {
            throw new IllegalArgumentException("hiddenBias must be finite, got " + hiddenBias);
        }

        this.sz = layerSizes.clone();
        this.lr = learningRate;
        this.batchSize = batchSize;
        this.loss = loss;
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
            double scale = init.scale(sz[l], sz[l + 1]);
            w[l] = new double[sz[l + 1]][sz[l]];
            for (int i = 0; i < sz[l + 1]; i++)
                for (int j = 0; j < sz[l]; j++)
                    w[l][i][j] = rnd.nextGaussian() * scale;

            b[l] = new double[sz[l + 1]];
            if (l < layers - 1) {
                Arrays.fill(b[l], hiddenBias);
            }

            dw[l] = new double[sz[l + 1]][sz[l]];
            db[l] = new double[sz[l + 1]];
            z[l] = new double[sz[l + 1]];
            delta[l] = new double[sz[l + 1]];
            a[l + 1] = new double[sz[l + 1]];
        }
    }

    protected abstract double activate(double z);

    protected abstract double derivative(double z, double activated);

    protected abstract String activationId();

    public final String id() {
        return activationId() + "/" + loss.id();
    }

    public final int[] layerSizes() {
        return sz.clone();
    }

    public final double[] parameters() {
        int count = 0;
        for (int l = 0; l < w.length; l++) {
            count += w[l].length * sz[l] + b[l].length;
        }
        double[] out = new double[count];
        int k = 0;
        for (int l = 0; l < w.length; l++) {
            for (double[] row : w[l]) {
                for (double v : row) {
                    out[k++] = v;
                }
            }
            for (double v : b[l]) {
                out[k++] = v;
            }
        }
        return out;
    }

    private void forward(double x) {
        a[0][0] = x;
        int last = sz.length - 2;
        for (int l = 0; l < last; l++) {
            double[][] wl = w[l];
            double[] al = a[l];
            double[] zl = z[l];
            double[] bl = b[l];
            double[] next = a[l + 1];
            for (int i = 0; i < zl.length; i++) {
                double[] row = wl[i];
                double acc = 0.0;
                for (int j = 0; j < row.length; j++) {
                    acc += row[j] * al[j];
                }
                double v = acc + bl[i];
                zl[i] = v;
                next[i] = activate(v);
            }
        }
        double[][] wo = w[last];
        double[] ao = a[last];
        double[] zo = z[last];
        double[] bo = b[last];
        double[] out = a[last + 1];
        for (int i = 0; i < zo.length; i++) {
            double[] row = wo[i];
            double acc = 0.0;
            for (int j = 0; j < row.length; j++) {
                acc += row[j] * ao[j];
            }
            double v = acc + bo[i];
            zo[i] = v;
            out[i] = v;
        }
    }

    @Override
    public final double predict(double x) {
        forward(x);
        return a[sz.length - 1][0];
    }

    @Override
    public final double train(double[] xs, double[] ys, int epochs) {
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

                    double predicted = a[sz.length - 1][0];
                    double residual = predicted - ys[idx];
                    epochSquaredError += residual * residual;
                    delta[layers - 1][0] = loss.gradient(predicted, ys[idx]);

                    for (int l = layers - 2; l >= 0; l--) {
                        double[] dl = delta[l];
                        double[] dNext = delta[l + 1];
                        double[][] wNext = w[l + 1];
                        double[] zl = z[l];
                        double[] activated = a[l + 1];
                        for (int j = 0; j < dl.length; j++) {
                            double acc = 0.0;
                            for (int k = 0; k < dNext.length; k++) {
                                acc += wNext[k][j] * dNext[k];
                            }
                            dl[j] = acc * derivative(zl[j], activated[j]);
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

    public final double meanSquaredError(double[] xs, double[] ys) {
        return mean(xs, ys, (predicted, target) -> {
            double e = predicted - target;
            return e * e;
        });
    }

    public final double meanLoss(double[] xs, double[] ys) {
        return mean(xs, ys, loss::value);
    }

    private double mean(double[] xs, double[] ys, DoubleBinaryOperator perSample) {
        if (xs.length != ys.length) {
            throw new IllegalArgumentException("xs and ys must have equal length, got "
                    + xs.length + " and " + ys.length);
        }
        if (xs.length == 0) return Double.NaN;
        double sum = 0.0;
        for (int i = 0; i < xs.length; i++) {
            sum += perSample.applyAsDouble(predict(xs[i]), ys[i]);
        }
        return sum / xs.length;
    }
}
