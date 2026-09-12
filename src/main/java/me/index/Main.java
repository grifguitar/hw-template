package me.index;

import me.index.ml.Net;
import me.index.view.Plot;

import java.util.Random;

public class Main {
    static void main(String[] args) {
        Context context = Context.read("config.properties", "");

        int n = context.keys.length;
        System.out.println(n);
        System.out.println(context.keys[0] + " " + context.keys[1] + " " + context.keys[2] + " ... "
                + context.keys[n - 2] + " " + context.keys[n - 1]);

        double minKey = context.keys[0];
        double maxKey = context.keys[n - 1];

        double[] xs = new double[n];
        double[] ys = new double[n];

        for (int i = 0; i < n; i++) {
            xs[i] = 2.0 * (context.keys[i] - minKey) / (maxKey - minKey) - 1.0;
            ys[i] = (double) i / (n - 1);
        }

        Net net = new Net(0.05, 32, new Random(42));
        net.train(xs, ys, 100);

        long maxErr = 0;
        long[] truePos = new long[n];
        long[] predPos = new long[n];

        for (int i = 0; i < n; i++) {
            truePos[i] = i;
            predPos[i] = Math.max(0, Math.min(n - 1, Math.round(net.predict(xs[i]) * (n - 1))));
            long err = Math.abs(predPos[i] - i);
            maxErr = Math.max(maxErr, err);
        }
        System.out.println("maxErr: " + maxErr);

        try {
            new Plot(900, 600)
                    .title("Keys - Positions")
                    .xlabel("keys")
                    .ylabel("positions")
                    .squareMarkers(context.keys.length >= 20_000)
                    .scatter(context.keys, truePos, Plot.BLUE, "true_positions", 0.1)
                    .scatter(context.keys, predPos, Plot.RED, "predicted_positions", 0.1)
                    .save("plot.pdf");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
