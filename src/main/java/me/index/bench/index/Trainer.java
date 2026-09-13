package me.index.bench.index;

import me.index.bench.data.Keys;
import me.index.config.parameters.enums.LossFunction;
import me.index.ml.Net;

import java.util.Objects;
import java.util.Random;

public final class Trainer {
    private Trainer() {
    }

    /**
     * @param learningRate {@code NaN} selects the default rate of the loss
     */
    public record Settings(int epochs, int sampleSize, int batchSize, LossFunction loss, double learningRate,
                           long seed) {
        public Settings {
            if (epochs <= 0) {
                throw new IllegalArgumentException("epochs must be positive, got " + epochs);
            }
            if (sampleSize < 2) {
                throw new IllegalArgumentException("the training sample needs at least 2 keys, got " + sampleSize);
            }
            if (batchSize <= 0) {
                throw new IllegalArgumentException("batch size must be positive, got " + batchSize);
            }
            Objects.requireNonNull(loss, "loss");
            if (!Double.isNaN(learningRate) && (!(learningRate > 0) || !Double.isFinite(learningRate))) {
                throw new IllegalArgumentException("learning rate must be finite and positive, got " + learningRate);
            }
        }

        public double effectiveLearningRate() {
            return Double.isNaN(learningRate) ? loss.defaultLearningRate() : learningRate;
        }
    }

    /**
     * Trains on up to {@code sampleSize} keys spread evenly by rank, then measures the error bounds on all keys.
     */
    public static NetModel train(long[] keys, NetSpec spec, Settings settings) {
        Keys.requireStrictlyIncreasing(keys);
        int n = keys.length;
        if (n < 2) {
            throw new IllegalArgumentException("a learned index needs at least 2 keys, got " + n);
        }
        KeyNormalizer normalizer = new KeyNormalizer(keys[0], keys[n - 1]);
        int m = Math.min(n, settings.sampleSize());
        double[] xs = new double[m];
        double[] ys = new double[m];
        for (int k = 0; k < m; k++) {
            int i = (m == n) ? k : (int) Math.round((double) k * (n - 1) / (m - 1));
            xs[k] = normalizer.normalize(keys[i]);
            ys[k] = (double) i / (n - 1);
        }

        LossFunction loss = settings.loss();
        Net net = spec.activation().create(spec.layerSizes(), settings.effectiveLearningRate(), settings.batchSize(),
                loss.create(loss.defaultParameter()), new Random(settings.seed()));
        long startedAt = System.nanoTime();
        net.train(xs, ys, settings.epochs());
        double seconds = (System.nanoTime() - startedAt) / 1e9;
        double mse = net.meanSquaredError(xs, ys);

        Kernel kernel = Kernel.of(spec.activation(), spec.layerSizes(), net.parameters()).scaleOutput(n - 1);
        NetIndex.Bounds bounds = NetIndex.bounds(keys, kernel);
        return new NetModel(spec, kernel.parameters(), n, keys[0], keys[n - 1], Keys.checksum(keys),
                bounds.errLo(), bounds.errHi(), bounds.meanAbsErr(),
                loss.name(), settings.epochs(), m, mse, seconds);
    }
}
