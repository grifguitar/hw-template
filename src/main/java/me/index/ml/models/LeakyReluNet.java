package me.index.ml.models;

import me.index.ml.Init;
import me.index.ml.Loss;
import me.index.ml.Net;

import java.util.Random;

public final class LeakyReluNet extends Net {
    public static final double DEFAULT_SLOPE = 0.01;

    private final double slope;

    public LeakyReluNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        this(layerSizes, learningRate, batchSize, loss, rnd, DEFAULT_SLOPE);
    }

    public LeakyReluNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd, double slope) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.HE, 0.0);
        this.slope = checkSlope(slope);
    }

    private static double checkSlope(double slope) {
        if (!(slope > 0) || !(slope < 1)) {
            throw new IllegalArgumentException("slope must lie in (0, 1), got " + slope);
        }
        return slope;
    }

    @Override
    protected double activate(double z) {
        return z > 0 ? z : slope * z;
    }

    @Override
    protected double derivative(double z, double activated) {
        return z > 0 ? 1.0 : slope;
    }

    @Override
    protected String activationId() {
        return "leaky_relu";
    }
}
