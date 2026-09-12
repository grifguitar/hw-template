package me.index.ml.models;

import me.index.ml.Init;
import me.index.ml.Loss;
import me.index.ml.Net;

import java.util.Random;

public final class SigmoidNet extends Net {
    public SigmoidNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.XAVIER, 0.0);
    }

    @Override
    protected double activate(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    protected double derivative(double z, double activated) {
        return activated * (1.0 - activated);
    }

    @Override
    protected String activationId() {
        return "sigmoid";
    }
}
