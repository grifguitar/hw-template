package me.index.ml.models;

import me.index.ml.Init;
import me.index.ml.Loss;
import me.index.ml.Net;

import java.util.Random;

public final class TanhNet extends Net {
    public TanhNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.XAVIER, 0.0);
    }

    @Override
    protected double activate(double z) {
        return Math.tanh(z);
    }

    @Override
    protected double derivative(double z, double activated) {
        return 1.0 - activated * activated;
    }

    @Override
    protected String activationId() {
        return "tanh";
    }
}
