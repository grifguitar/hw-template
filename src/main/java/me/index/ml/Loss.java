package me.index.ml;

public interface Loss {
    double value(double predicted, double target);

    double gradient(double predicted, double target);

    String id();
}
