package me.index.ml;

import me.index.Pair;

import java.util.Random;

import static me.index.math.Maths.*;

public class Net {
    private final double[][][] W;
    private final double[][][] B;
    private final int[] sz;
    private final double lr;
    private final int batchsize;
    private final Random rnd;

    Net(double learningRate, int batchsize) {
        this.sz = new int[]{1, 4, 4, 1};
        this.lr = learningRate;
        this.batchsize = batchsize;
        this.rnd = new Random(42);

        W = new double[sz.length - 1][][];
        B = new double[sz.length - 1][][];

        for (int l = 0; l < sz.length - 1; l++) {
            W[l] = new double[sz[l + 1]][sz[l]];
            for (int i = 0; i < sz[l + 1]; i++)
                for (int j = 0; j < sz[l]; j++)
                    W[l][i][j] = rnd.nextGaussian() * Math.sqrt(2.0 / sz[l]);

            B[l] = new double[sz[l + 1]][1];
            for (int i = 0; i < sz[l + 1]; i++)
                for (int j = 0; j < 1; j++)
                    B[l][i][j] = 0.3;
        }

        B[sz.length - 1 - 1] = new double[sz[sz.length - 1]][1];
    }

    private Pair<double[][][], double[][][]> forward(double[][] x) {
        double[][][] z = new double[sz.length - 1][][];
        double[][][] a = new double[sz.length][][];
        a[0] = x;
        for (int l = 0; l < sz.length - 1; l++) {
            z[l] = sum(mul(W[l], a[l]), B[l]);
            a[l + 1] = (l < sz.length - 2)
                    ? apply(z[l], (t) -> (Math.max(0.0, t)))
                    : z[l];
        }
        return new Pair<>(z, a);
    }

    double predict(double x) {
        return forward(new double[][]{{x}}).second[sz.length - 1][0][0];
    }

    void train(double[] xs, double[] ys, int epochs) {
        int n = xs.length;

        int[] perm = new int[n];
        for (int i = 0; i < n; i++)
            perm[i] = i;

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int i = perm.length - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }

            for (int start = 0; start < n; start += batchsize) {
                int end = Math.min(start + batchsize, n);

                double[] xb = new double[end - start];
                double[] yb = new double[end - start];

                for (int i = 0; i < (end - start); i++) {
                    xb[i] = xs[perm[start + i]];
                    yb[i] = ys[perm[start + i]];
                }

                double[][][] sumDW = new double[sz.length - 1][][];
                double[][][] sumDB = new double[sz.length - 1][][];
                for (int l = 0; l < sz.length - 1; l++) {
                    sumDW[l] = new double[sz[l + 1]][sz[l]];
                    sumDB[l] = new double[sz[l + 1]][1];
                }

                for (int i = 0; i < xb.length; i++) {
                    Pair<double[][][], double[][][]> c = forward(new double[][]{{xb[i]}});
                    double[][][] z = c.first;
                    double[][][] a = c.second;

                    double[][] target = {{yb[i]}};

                    double[][][] delta = new double[sz.length - 1][][];
                    delta[sz.length - 2] = sub(a[sz.length - 1], target);
                    for (int l = sz.length - 3; l >= 0; l--) {
                        delta[l] = dot(
                                mul(tp(W[l + 1]), delta[l + 1]),
                                apply(z[l], (t) -> (t > 0 ? 1.0 : 0.0))
                        );
                    }

                    for (int l = 0; l < sz.length - 1; l++) {
                        sumDW[l] = sum(sumDW[l], mul(delta[l], tp(a[l])));
                        sumDB[l] = sum(sumDB[l], delta[l]);
                    }
                }

                double step = lr / xb.length;
                for (int l = 0; l < sz.length - 1; l++) {
                    W[l] = sub(W[l], mul(sumDW[l], step));
                    B[l] = sub(B[l], mul(sumDB[l], step));
                }
            }
        }
    }
}
