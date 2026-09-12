package me.index.ml.models;

import me.index.ml.Init;
import me.index.ml.Loss;
import me.index.ml.Net;

import java.util.Random;

public final class SoftsignNet extends Net {
    public SoftsignNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.XAVIER, 0.0);
    }

    @Override
    protected double activate(double z) {
        return z / (1.0 + Math.abs(z));
    }

    @Override
    protected double derivative(double z, double activated) {
        double d = 1.0 + Math.abs(z);
        return 1.0 / (d * d);
    }

    @Override
    protected String activationId() {
        return "softsign";
    }
}
