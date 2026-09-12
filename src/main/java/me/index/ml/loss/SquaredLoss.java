package me.index.ml.loss;

import me.index.ml.Loss;

public final class SquaredLoss implements Loss {
    @Override
    public double value(double predicted, double target) {
        double e = predicted - target;
        return 0.5 * e * e;
    }

    @Override
    public double gradient(double predicted, double target) {
        return predicted - target;
    }

    @Override
    public String id() {
        return "squared";
    }
}
