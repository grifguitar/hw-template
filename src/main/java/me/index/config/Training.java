package me.index.config;

import java.util.List;

public record Training(
        long seed,
        double learningRate,
        int batchSize,
        int epochs,
        List<Integer> hiddenLayers,
        String plotPath
) {
    public static final long DEFAULT_SEED = 42L;
    public static final double DEFAULT_LEARNING_RATE = 0.05;
    public static final int DEFAULT_BATCH_SIZE = 32;
    public static final int DEFAULT_EPOCHS = 100;
    public static final List<Integer> DEFAULT_HIDDEN_LAYERS = List.of(4, 4);
    public static final String DEFAULT_PLOT_PATH = "plot.pdf";

    public Training {
        hiddenLayers = List.copyOf(hiddenLayers);
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
