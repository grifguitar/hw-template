package me.index.ml;

import me.index.ml.loss.AbsoluteLoss;
import me.index.ml.loss.HuberLoss;
import me.index.ml.loss.LogCoshLoss;
import me.index.ml.loss.PowerLoss;
import me.index.ml.loss.SquaredLoss;
import me.index.ml.models.LeakyReluNet;
import me.index.ml.models.SigmoidNet;
import me.index.ml.models.SoftsignNet;
import me.index.ml.models.TanhNet;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BackpropTest {
    private static final int[] SHAPE = {1, 4, 3, 1};
    private static final int SAMPLE = 16;
    private static final int BATCH = SAMPLE;
    private static final double LR = 1e-4;
    private static final double H = 1e-6;
    private static final double ABSOLUTE_NOISE = 1e-8;
    private static final double RELATIVE_TOLERANCE = 1e-4;

    private static final List<BiFunction<Loss, Random, Net>> NETS = List.of(
            (loss, rnd) -> new TanhNet(SHAPE, LR, BATCH, loss, rnd),
            (loss, rnd) -> new SigmoidNet(SHAPE, LR, BATCH, loss, rnd),
            (loss, rnd) -> new SoftsignNet(SHAPE, LR, BATCH, loss, rnd),
            (loss, rnd) -> new LeakyReluNet(SHAPE, LR, BATCH, loss, rnd));

    private static final List<Loss> LOSSES = List.of(
            new SquaredLoss(), new AbsoluteLoss(), new HuberLoss(0.05),
            new LogCoshLoss(0.05), new PowerLoss(3, 0.1), new PowerLoss(3));

    @Test
    void everyLossAndActivationAgreesWithFiniteDifferences() throws Exception {
        double[][] sample = sample(SAMPLE);
        for (Loss loss : LOSSES) {
            for (BiFunction<Loss, Random, Net> factory : NETS) {
                checkOneNet(factory.apply(loss, new Random(1)), loss, sample[0], sample[1]);
            }
        }
    }

    private void checkOneNet(Net net, Loss loss, double[] xs, double[] ys) throws Exception {
        double[][][] w = field(net, "w");
        double[][] b = field(net, "b");

        double[][][] before = copy(w);
        double[][] beforeB = copy(b);
        net.train(xs, ys, 1);
        double[][][] after = copy(w);
        double[][] afterB = copy(b);
        restore(before, w);
        restore(beforeB, b);

        for (int l = 0; l < w.length; l++) {
            for (int i = 0; i < w[l].length; i++) {
                for (int j = 0; j < w[l][i].length; j++) {
                    double applied = (before[l][i][j] - after[l][i][j]) / LR;
                    assertMatches(applied, numeric(net, loss, xs, ys, w[l][i], j),
                            net.id() + " w[" + l + "][" + i + "][" + j + "]");
                }
            }
            for (int i = 0; i < b[l].length; i++) {
                double applied = (beforeB[l][i] - afterB[l][i]) / LR;
                assertMatches(applied, numeric(net, loss, xs, ys, b[l], i),
                        net.id() + " b[" + l + "][" + i + "]");
            }
        }
    }

    private static double numeric(Net net, Loss loss, double[] xs, double[] ys, double[] cell, int index) {
        double saved = cell[index];
        cell[index] = saved + H;
        double up = meanLoss(net, loss, xs, ys);
        cell[index] = saved - H;
        double down = meanLoss(net, loss, xs, ys);
        cell[index] = saved;
        return (up - down) / (2 * H);
    }

    private static double meanLoss(Net net, Loss loss, double[] xs, double[] ys) {
        double sum = 0.0;
        for (int i = 0; i < xs.length; i++) {
            sum += loss.value(net.predict(xs[i]), ys[i]);
        }
        return sum / xs.length;
    }

    private static void assertMatches(double applied, double numeric, String where) {
        double absolute = Math.abs(applied - numeric);
        if (absolute < ABSOLUTE_NOISE) return;
        double relative = absolute / (Math.abs(applied) + Math.abs(numeric));
        assertTrue(relative < RELATIVE_TOLERANCE,
                where + ": backprop " + applied + " vs numeric " + numeric);
    }

    private static double[][] sample(int n) {
        double[] xs = new double[n];
        double[] ys = new double[n];
        Random noise = new Random(7);
        for (int i = 0; i < n; i++) {
            xs[i] = 2.0 * i / (n - 1) - 1.0;
            ys[i] = 0.5 + 0.3 * Math.sin(3 * xs[i]) + 0.01 * noise.nextGaussian();
        }
        return new double[][]{xs, ys};
    }

    @SuppressWarnings("unchecked")
    private static <T> T field(Net net, String name) throws Exception {
        Field f = Net.class.getDeclaredField(name);
        f.setAccessible(true);
        return (T) f.get(net);
    }

    private static double[][][] copy(double[][][] src) {
        double[][][] out = new double[src.length][][];
        for (int i = 0; i < src.length; i++) {
            out[i] = copy(src[i]);
        }
        return out;
    }

    private static double[][] copy(double[][] src) {
        double[][] out = new double[src.length][];
        for (int i = 0; i < src.length; i++) {
            out[i] = src[i].clone();
        }
        return out;
    }

    private static void restore(double[][][] saved, double[][][] target) {
        for (int i = 0; i < saved.length; i++) {
            restore(saved[i], target[i]);
        }
    }

    private static void restore(double[][] saved, double[][] target) {
        for (int i = 0; i < saved.length; i++) {
            System.arraycopy(saved[i], 0, target[i], 0, saved[i].length);
        }
    }
}
