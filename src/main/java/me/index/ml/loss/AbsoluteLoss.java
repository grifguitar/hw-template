package me.index.ml.loss;

import me.index.ml.Loss;

public final class AbsoluteLoss implements Loss {
    @Override
    public double value(double predicted, double target) {
        return Math.abs(predicted - target);
    }

    @Override
    public double gradient(double predicted, double target) {
        return Math.signum(predicted - target);
    }

    @Override
    public String id() {
        return "absolute";
    }
}
