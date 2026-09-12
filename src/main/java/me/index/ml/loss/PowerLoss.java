package me.index.ml.loss;

import me.index.ml.Loss;

public final class PowerLoss implements Loss {
    public static final double DEFAULT_SCALE = 1.0;

    private final double p;
    private final double scale;

    public PowerLoss(double p) {
        this(p, DEFAULT_SCALE);
    }

    public PowerLoss(double p, double scale) {
        if (!(p >= 1) || !Double.isFinite(p)) {
            throw new IllegalArgumentException("p must be finite and at least 1, got " + p);
        }
        if (!(scale > 0) || !Double.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and positive, got " + scale);
        }
        this.p = p;
        this.scale = scale;
    }

    @Override
    public double value(double predicted, double target) {
        double u = Math.abs(predicted - target) / scale;
        return scale * scale / p * Math.pow(u, p);
    }

    @Override
    public double gradient(double predicted, double target) {
        double e = predicted - target;
        double u = Math.abs(e) / scale;
        return scale * Math.pow(u, p - 1) * Math.signum(e);
    }

    @Override
    public String id() {
        return "power(" + p + "," + scale + ")";
    }
}
