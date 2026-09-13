package me.index.bench.index;

import me.index.config.parameters.enums.Activation;
import me.index.ml.Net;
import me.index.ml.loss.SquaredLoss;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class KernelTest {
    private static final int[][] SHAPES = {{1, 1}, {1, 8, 1}, {1, 5, 3, 1}, {1, 2, 7, 4, 1}};

    private static Net trainedNet(Activation activation, int[] shape) {
        Net net = activation.create(shape, 0.01, 16, new SquaredLoss(), new Random(7));
        double[] xs = new double[200];
        double[] ys = new double[200];
        for (int i = 0; i < xs.length; i++) {
            xs[i] = 2.0 * i / (xs.length - 1) - 1.0;
            ys[i] = (xs[i] * xs[i] * xs[i] + 1) / 2;
        }
        net.train(xs, ys, 3);
        return net;
    }

    @Test
    void everyActivationPredictsExactlyLikeTheNetItWasExportedFrom() {
        for (Activation activation : Activation.values()) {
            for (int[] shape : SHAPES) {
                Net net = trainedNet(activation, shape);
                Kernel kernel = Kernel.of(activation, shape, net.parameters());
                assertEquals(activation, kernel.activation());
                for (int i = 0; i <= 1000; i++) {
                    double x = -1.5 + 3.0 * i / 1000;
                    assertEquals(net.predict(x), kernel.predict(x),
                            () -> activation + " " + Arrays.toString(shape) + " x=" + x);
                }
            }
        }
    }

    @Test
    void scaleOutputMultipliesOnlyTheOutputLayer() {
        int[] shape = {1, 5, 3, 1};
        Net net = trainedNet(Activation._tanh, shape);
        Kernel kernel = Kernel.of(Activation._tanh, shape, net.parameters());
        Kernel scaled = kernel.scaleOutput(1000);

        double[] before = kernel.parameters();
        double[] after = scaled.parameters();
        int outputLayer = before.length - 3 - 1;
        for (int i = 0; i < before.length; i++) {
            assertEquals(i < outputLayer ? before[i] : before[i] * 1000, after[i], "parameter " + i);
        }
        for (double x = -1; x <= 1; x += 0.01) {
            double expected = kernel.predict(x) * 1000;
            assertEquals(expected, scaled.predict(x), 1e-9 * Math.max(1, Math.abs(expected)));
        }
        assertArrayEquals(before, kernel.parameters(), "scaling must not touch the original kernel");
    }

    @Test
    void copiesPredictTheSameAndDoNotShareBuffers() {
        int[] shape = {1, 6, 6, 1};
        Kernel kernel = Kernel.of(Activation._softsign, shape, trainedNet(Activation._softsign, shape).parameters());
        Kernel copy = kernel.copy();
        double[] reference = new double[100];
        for (int i = 0; i < reference.length; i++) {
            reference[i] = kernel.predict(i / 50.0 - 1);
        }
        for (int i = 0; i < reference.length; i++) {
            double x = i / 50.0 - 1;
            assertEquals(reference[i], copy.predict(x));
            assertEquals(reference[i], kernel.predict(x));
        }
        assertEquals(Activation._softsign, copy.activation());
    }

    @Test
    void parameterCountAndMemoryFollowTheLayout() {
        assertEquals(2, Kernel.parameterCount(new int[]{1, 1}));
        assertEquals(33, Kernel.parameterCount(new int[]{1, 4, 4, 1}));
        assertEquals(2 + 16 + 9, Kernel.parameterCount(new int[]{1, 1, 8, 1}));

        int[] shape = {1, 4, 4, 1};
        Kernel kernel = Kernel.of(Activation._relu, shape, new double[33]);
        assertEquals(8L * (33 + 4 + 4) + 4L * 4, kernel.sizeBytes());

        double[] params = kernel.parameters();
        params[0] = 99;
        assertEquals(0.0, kernel.parameters()[0], "parameters() must return a copy");
        int[] sizes = kernel.layerSizes();
        sizes[1] = 99;
        assertArrayEquals(shape, kernel.layerSizes(), "layerSizes() must return a copy");
    }

    @Test
    void theNetExportsItsParametersInKernelOrder() {
        Net net = trainedNet(Activation._relu, new int[]{1, 2, 1});
        double[] p = net.parameters();
        assertEquals(7, p.length);
        double x = 0.3;
        double h0 = Math.max(0, p[0] * x + p[2]);
        double h1 = Math.max(0, p[1] * x + p[3]);
        assertEquals(p[4] * h0 + p[5] * h1 + p[6], net.predict(x), 1e-12);
    }

    @Test
    void invalidShapesAndParameterCountsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Kernel.of(Activation._relu, new int[]{1}, new double[0]));
        assertThrows(IllegalArgumentException.class, () -> Kernel.of(Activation._relu, new int[]{2, 1}, new double[3]));
        assertThrows(IllegalArgumentException.class, () -> Kernel.of(Activation._relu, new int[]{1, 2}, new double[4]));
        assertThrows(IllegalArgumentException.class, () -> Kernel.of(Activation._relu, new int[]{1, 0, 1}, new double[1]));
        assertThrows(IllegalArgumentException.class, () -> Kernel.of(Activation._relu, new int[]{1, 4, 1}, new double[12]));
    }
}
