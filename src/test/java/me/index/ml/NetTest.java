package me.index.ml;

import me.index.config.LossFunction;
import me.index.ml.loss.PowerLoss;
import me.index.ml.loss.SquaredLoss;
import me.index.ml.models.ReluNet;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class NetTest {
    private static final int[] SHAPE = {1, 8, 8, 1};

    private static double[][] linearSample(int n) {
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = 2.0 * i / (n - 1) - 1.0;
            ys[i] = (double) i / (n - 1);
        }
        return new double[][]{xs, ys};
    }

    @Test
    void trainingReducesTheError() {
        double[][] s = linearSample(500);
        Net net = new ReluNet(SHAPE, 0.05, 32, new SquaredLoss(), new Random(1));
        double before = net.train(s[0], s[1], 1);
        double after = net.train(s[0], s[1], 200);
        assertTrue(after < before / 10, "MSE " + before + " -> " + after);
        assertTrue(after < 1e-3, "MSE after training: " + after);
    }

    @Test
    void fitsALinearTargetCloselyEnough() {
        double[][] s = linearSample(500);
        Net net = new ReluNet(SHAPE, 0.05, 32, new SquaredLoss(), new Random(1));
        net.train(s[0], s[1], 300);
        for (int i = 0; i < s[0].length; i += 25) {
            assertEquals(s[1][i], net.predict(s[0][i]), 0.05, "x = " + s[0][i]);
        }
    }

    @Test
    void sameSeedGivesIdenticalPredictions() {
        double[][] s = linearSample(200);
        Net a = new ReluNet(SHAPE, 0.05, 16, new SquaredLoss(), new Random(42));
        Net b = new ReluNet(SHAPE, 0.05, 16, new SquaredLoss(), new Random(42));
        assertEquals(a.train(s[0], s[1], 20), b.train(s[0], s[1], 20));
        for (double x : s[0]) {
            assertEquals(a.predict(x), b.predict(x));
        }
    }

    @Test
    void predictIsStableWhenCalledRepeatedly() {
        Net net = new ReluNet(SHAPE, 0.05, 16, new SquaredLoss(), new Random(5));
        double first = net.predict(0.25);
        assertEquals(first, net.predict(0.25));
        net.predict(-1.0);
        assertEquals(first, net.predict(0.25));
    }

    @Test
    void degenerateShapeWithoutHiddenLayersIsLinearRegression() {
        double[][] s = linearSample(200);
        Net net = new ReluNet(new int[]{1, 1}, 0.05, 16, new SquaredLoss(), new Random(2));
        net.train(s[0], s[1], 300);
        assertEquals(0.5, net.predict(0.0), 0.02);
        assertEquals(1.0, net.predict(1.0), 0.05);
    }

    @Test
    void invalidArgumentsAreRejected() {
        Random rnd = new Random(1);
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(new int[]{1}, 0.05, 8, new SquaredLoss(), rnd));
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(new int[]{2, 4, 1}, 0.05, 8, new SquaredLoss(), rnd));
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(new int[]{1, 4, 2}, 0.05, 8, new SquaredLoss(), rnd));
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(new int[]{1, 0, 1}, 0.05, 8, new SquaredLoss(), rnd));
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(SHAPE, 0.0, 8, new SquaredLoss(), rnd));
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(SHAPE, Double.NaN, 8, new SquaredLoss(), rnd));
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(SHAPE, 0.05, 0, new SquaredLoss(), rnd));
        assertThrows(IllegalArgumentException.class, () -> new ReluNet(SHAPE, 0.05, 8, new SquaredLoss(), null));

        Net net = new ReluNet(SHAPE, 0.05, 8, new SquaredLoss(), rnd);
        assertThrows(IllegalArgumentException.class, () -> net.train(new double[3], new double[2], 1));
        assertThrows(IllegalArgumentException.class, () -> net.train(new double[3], new double[3], -1));
    }

    @Test
    void reportedMetricsAreExactAndAgreeWithTheLoss() {
        double[][] s = linearSample(300);
        Net net = new ReluNet(SHAPE, 0.05, 32, new SquaredLoss(), new Random(1));
        net.train(s[0], s[1], 100);

        double mse = net.meanSquaredError(s[0], s[1]);
        double sum = 0.0;
        for (int i = 0; i < s[0].length; i++) {
            double e = net.predict(s[0][i]) - s[1][i];
            sum += e * e;
        }
        assertEquals(sum / s[0].length, mse, 1e-15, "meanSquaredError must be measured, not accumulated");
        assertEquals(0.5 * mse, net.meanLoss(s[0], s[1]), 1e-15);

        assertTrue(Double.isNaN(net.meanSquaredError(new double[0], new double[0])));
        assertThrows(IllegalArgumentException.class, () -> net.meanLoss(new double[3], new double[2]));
    }

    @Test
    void theReturnOfTrainIsARunningEstimateOfTheExactError() {
        double[][] s = linearSample(300);
        Net net = new ReluNet(SHAPE, 0.05, 32, new SquaredLoss(), new Random(1));
        double running = net.train(s[0], s[1], 200);
        double exact = net.meanSquaredError(s[0], s[1]);
        assertTrue(exact <= running, "the exact error must not exceed the running one: " + exact + " vs " + running);
        assertEquals(running, exact, running);
    }

    @Test
    void everyConfiguredLossTrainsWithoutDiverging() {
        double[][] s = linearSample(300);
        for (LossFunction lf : LossFunction.values()) {
            Loss loss = lf.create(lf.defaultParameter());
            Net net = new ReluNet(SHAPE, 0.05, 32, loss, new Random(1));
            net.train(s[0], s[1], 200);
            double mse = net.meanSquaredError(s[0], s[1]);
            assertTrue(Double.isFinite(mse), loss.id() + " diverged: " + mse);
            assertTrue(mse < 5e-3, loss.id() + " failed to fit a line: MSE " + mse);
        }
    }

    @Test
    void aSteepPowerLossStaysStableOnNormalisedTargets() {
        double[][] s = linearSample(300);
        for (double p : new double[]{1.5, 3.0, 4.0, 6.0}) {
            Net net = new ReluNet(SHAPE, 0.05, 32, new PowerLoss(p), new Random(1));
            double mse = net.train(s[0], s[1], 200);
            assertTrue(Double.isFinite(mse), "power(" + p + ") diverged: " + mse);
        }
    }

    @Test
    void constructorCopiesTheLayerArray() {
        int[] shape = {1, 4, 1};
        Net net = new ReluNet(shape, 0.05, 8, new SquaredLoss(), new Random(1));
        shape[1] = 999;
        assertDoesNotThrow(() -> net.predict(0.5));
    }

    @Test
    void asymmetricAndDeepShapesTrainJustAsWell() {
        double[][] s = linearSample(400);
        int[][] shapes = {{1, 3, 7, 1}, {1, 7, 3, 1}, {1, 2, 9, 4, 1}, {1, 16, 1}, {1, 5, 5, 5, 1}};
        for (int[] shape : shapes) {
            Net net = new ReluNet(shape, 0.05, 32, new SquaredLoss(), new Random(3));
            double before = net.train(s[0], s[1], 1);
            double after = net.train(s[0], s[1], 200);
            String name = java.util.Arrays.toString(shape);
            assertTrue(Double.isFinite(after), name + " diverged: " + after);
            assertTrue(after < before, name + ": MSE " + before + " -> " + after);
            assertTrue(after < 5e-3, name + ": MSE after training " + after);
            assertEquals(0.5, net.predict(0.0), 0.08, name);
        }
    }

    @Test
    void fitsANonLinearTargetWithEnoughNeurons() {
        int n = 400;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = 2.0 * i / (n - 1) - 1.0;
            ys[i] = (xs[i] * xs[i] * xs[i] + 1) / 2;
        }
        Net net = new ReluNet(new int[]{1, 12, 8, 1}, 0.05, 32, new SquaredLoss(), new Random(4));
        double mse = net.train(xs, ys, 400);
        assertTrue(mse < 1e-3, "MSE on a cubic target: " + mse);
        for (int i = 0; i < n; i += 40) {
            assertEquals(ys[i], net.predict(xs[i]), 0.1, "x = " + xs[i]);
        }
    }

    @Test
    void divergenceIsVisibleInTheReturnedLoss() {
        double[][] s = linearSample(100);
        Net net = new ReluNet(SHAPE, 1e9, 8, new SquaredLoss(), new Random(1));
        assertFalse(Double.isFinite(net.train(s[0], s[1], 20)),
                "an absurd learning rate must surface as a non-finite MSE");
    }

    @Test
    void zeroEpochsOrEmptySampleIsANoOp() {
        double[][] s = linearSample(50);
        Net net = new ReluNet(SHAPE, 0.05, 8, new SquaredLoss(), new Random(1));
        double before = net.predict(0.5);
        assertTrue(Double.isNaN(net.train(s[0], s[1], 0)));
        assertTrue(Double.isNaN(net.train(new double[0], new double[0], 10)));
        assertEquals(before, net.predict(0.5), "no training must leave the weights untouched");
    }
}
