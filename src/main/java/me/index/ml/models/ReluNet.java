package me.index.ml.models;

import me.index.ml.Init;
import me.index.ml.Loss;
import me.index.ml.Net;

import java.util.Random;

public final class ReluNet extends Net {
    public static final double INITIAL_BIAS = 0.3;

    public ReluNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.HE, INITIAL_BIAS);
    }

    @Override
    protected double activate(double z) {
        return Math.max(0.0, z);
    }

    @Override
    protected double derivative(double z, double activated) {
        return z > 0 ? 1.0 : 0.0;
    }

    @Override
    protected String activationId() {
        return "relu";
    }
}
