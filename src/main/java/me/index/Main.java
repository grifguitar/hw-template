package me.index;

import me.index.ml.Net;
import me.index.view.Plot;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

public class Main {
    private static final String DEFAULT_CONFIG = "config.properties";
    private static final int PREVIEW = 3;

    static void main(String[] args) {
        if (args.length > 2) {
            System.err.println("usage: Main [config.properties] [sosd-data-dir]");
            System.exit(2);
        }
        Path configFile = Path.of(args.length > 0 ? args[0] : DEFAULT_CONFIG);
        Path dataDir = Path.of(args.length > 1 ? args[1] : "");

        try {
            run(configFile, dataDir);
        } catch (IOException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("error: " + e.getMessage());
            System.exit(1);
        }
    }

    static void run(Path configFile, Path dataDir) throws IOException {
        Context context = Context.read(configFile, dataDir);

        long[] keys = context.keys;
        int n = keys.length;
        System.out.println("keyset=" + context.config.keyset() + " data.size=" + context.config.dataSize()
                + " target max.err=" + context.config.maxErr().value);
        System.out.println("distinct keys: " + n);
        System.out.println(preview(keys));

        double minKey = keys[0];
        double maxKey = keys[n - 1];
        double keySpan = maxKey - minKey;

        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = 2.0 * (keys[i] - minKey) / keySpan - 1.0;
            ys[i] = (double) i / (n - 1);
        }

        Net net = new Net(
                context.config.training().layerSizes(),
                context.config.training().learningRate(),
                context.config.training().batchSize(),
                new Random(context.config.training().seed())
        );
        long startedAt = System.nanoTime();
        double mse = net.train(xs, ys, context.config.training().epochs());
        System.out.printf(Locale.US, "trained %d epoch(s) in %.1f s, final MSE %.3e%n",
                context.config.training().epochs(), (System.nanoTime() - startedAt) / 1e9, mse);
        if (!Double.isFinite(mse)) {
            throw new IllegalStateException("training diverged (MSE = " + mse
                    + "); lower train.learning.rate or shrink net.hidden.layers");
        }

        long maxErr = 0;
        long[] truePos = new long[n];
        long[] predPos = new long[n];
        for (int i = 0; i < n; i++) {
            truePos[i] = i;
            predPos[i] = Math.max(0, Math.min(n - 1, Math.round(net.predict(xs[i]) * (n - 1))));
            maxErr = Math.max(maxErr, Math.abs(predPos[i] - i));
        }
        System.out.println("maxErr: " + maxErr);

        new Plot(900, 600)
                .title("Keys - Positions")
                .xlabel("keys")
                .ylabel("positions")
                .squareMarkers(n >= 20_000)
                .scatter(keys, truePos, Plot.BLUE, "true_positions", 0.1)
                .scatter(keys, predPos, Plot.RED, "predicted_positions", 0.1)
                .save(context.config.training().plotPath());
        System.out.println("plot written to " + Path.of(context.config.training().plotPath()).toAbsolutePath());
    }

    static String preview(long[] keys) {
        StringBuilder sb = new StringBuilder();
        if (keys.length <= 2 * PREVIEW) {
            for (int i = 0; i < keys.length; i++) {
                sb.append(i > 0 ? " " : "").append(keys[i]);
            }
            return sb.toString();
        }
        for (int i = 0; i < PREVIEW; i++) {
            sb.append(keys[i]).append(' ');
        }
        sb.append("...");
        for (int i = keys.length - PREVIEW; i < keys.length; i++) {
            sb.append(' ').append(keys[i]);
        }
        return sb.toString();
    }
}
