package me.index.ml.loss;

import me.index.ml.Loss;

public final class HuberLoss implements Loss {
    private final double delta;

    public HuberLoss(double delta) {
        if (!(delta > 0) || !Double.isFinite(delta)) {
            throw new IllegalArgumentException("delta must be finite and positive, got " + delta);
        }
        this.delta = delta;
    }

    @Override
    public double value(double predicted, double target) {
        double e = Math.abs(predicted - target);
        return (e <= delta) ? 0.5 * e * e : delta * (e - 0.5 * delta);
    }

    @Override
    public double gradient(double predicted, double target) {
        double e = predicted - target;
        return (Math.abs(e) <= delta) ? e : delta * Math.signum(e);
    }

    @Override
    public String id() {
        return "huber(" + delta + ")";
    }
}
