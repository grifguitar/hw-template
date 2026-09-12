package me.index.ml.loss;

import me.index.ml.Loss;

public final class LogCoshLoss implements Loss {
    private final double scale;

    public LogCoshLoss(double scale) {
        if (!(scale > 0) || !Double.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and positive, got " + scale);
        }
        this.scale = scale;
    }

    @Override
    public double value(double predicted, double target) {
        double u = Math.abs(predicted - target) / scale;
        return scale * scale * (u + Math.log1p(Math.exp(-2.0 * u)) - Math.log(2.0));
    }

    @Override
    public double gradient(double predicted, double target) {
        return scale * Math.tanh((predicted - target) / scale);
    }

    @Override
    public String id() {
        return "logcosh(" + scale + ")";
    }
}
