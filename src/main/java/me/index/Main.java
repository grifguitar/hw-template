package me.index;

import me.index.ml.Net;

import java.io.IOException;
import java.util.Random;

import static me.index.view.Plot.plot;

public class LearnedIndex {

    static void main(String[] args) throws IOException {
        Context context = Context.read("config.properties", "");

        int n = context.keys.length;

        long minKey = context.keys[0];
        long maxKey = context.keys[n - 1];

        double[] xs = new double[n];
        double[] ys = new double[n];

        for (int i = 0; i < n; i++) {
            xs[i] = 2.0 * (context.keys[i] - minKey) / (maxKey - minKey) - 1.0;
            ys[i] = (double) i / (n - 1);
        }

        Net net = new Net(0.05, 32, new Random(42));
        net.train(xs, ys, 200);

        long maxErr = 0;
        long[] pred = new long[n];

        for (int i = 0; i < n; i++) {
            pred[i] = Math.max(0, Math.min(n - 1, Math.round(net.predict(xs[i]) * (n - 1))));
            long err = Math.abs(pred[i] - i);
            maxErr = Math.max(maxErr, err);
            if (i % 1000 == 0)
                System.out.println(context.keys[i] + " " + i + " " + pred[i] + " " + err);
        }

        plot(context.keys, pred, "plot.png");
    }
}
