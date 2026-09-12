package me.index.ml;

public interface Model {
    double predict(double x);

    double train(double[] xs, double[] ys, int epochs);
}
