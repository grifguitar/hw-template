package me.index.config;

import me.index.ml.Net;

import java.util.List;
import java.util.Random;

public record Training(
        long seed,
        double learningRate,
        int batchSize,
        int epochs,
        List<Integer> hiddenLayers,
        Activation activation,
        LossFunction lossFunction,
        double lossParameter,
        String plotPath
) {
    public static final long DEFAULT_SEED = 42L;
    public static final int DEFAULT_BATCH_SIZE = 32;
    public static final int DEFAULT_EPOCHS = 100;
    public static final List<Integer> DEFAULT_HIDDEN_LAYERS = List.of(4, 4);
    public static final Activation DEFAULT_ACTIVATION = Activation._relu;
    public static final LossFunction DEFAULT_LOSS = LossFunction._squared;
    public static final String DEFAULT_PLOT_PATH = "plot.pdf";

    public Training {
        hiddenLayers = List.copyOf(hiddenLayers);
        if (activation == null) throw new IllegalArgumentException("activation must not be null");
        if (lossFunction == null) throw new IllegalArgumentException("lossFunction must not be null");
        lossFunction.create(lossParameter);
    }

    public Net newNet(Random rnd) {
        return activation.create(layerSizes(), learningRate, batchSize, lossFunction.create(lossParameter), rnd);
    }

    public int[] layerSizes() {
        int[] sz = new int[hiddenLayers.size() + 2];
        sz[0] = 1;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            sz[i + 1] = hiddenLayers.get(i);
        }
        sz[sz.length - 1] = 1;
        return sz;
    }
}
