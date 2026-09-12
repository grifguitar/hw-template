package me.index.ml;

import me.index.ml.loss.*;
import me.index.ml.models.*;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.*;

class ActivationTest {

    private static final int[] SHAPE = {1, 6, 6, 1};

    private static List<Net> all(Loss loss, double lr, long seed) {
        return List.of(
                new ReluNet(SHAPE, lr, 32, loss, new Random(seed)),
                new LeakyReluNet(SHAPE, lr, 32, loss, new Random(seed)),
                new SigmoidNet(SHAPE, lr, 32, loss, new Random(seed)),
                new TanhNet(SHAPE, lr, 32, loss, new Random(seed)),
                new SoftsignNet(SHAPE, lr, 32, loss, new Random(seed)));
    }

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
    void derivativeMatchesTheNumericalSlopeOfTheActivation() {
        double h = 1e-6;
        for (Net net : all(new SquaredLoss(), 0.05, 1)) {
            for (double z = -4.0; z <= 4.0; z += 0.037) {
                double numeric = (net.activate(z + h) - net.activate(z - h)) / (2 * h);
                double analytic = net.derivative(z, net.activate(z));
                assertEquals(numeric, analytic, 1e-4,
                        net.id() + " at z = " + z);
            }
        }
    }

    @Test
    void activationsHaveTheExpectedShape() {
        Loss loss = new SquaredLoss();
        Random r = new Random(1);
        Net relu = new ReluNet(SHAPE, 0.05, 32, loss, r);
        assertEquals(0.0, relu.activate(-5));
        assertEquals(3.0, relu.activate(3));

        Net leaky = new LeakyReluNet(SHAPE, 0.05, 32, loss, r);
        assertEquals(-0.05, leaky.activate(-5), 1e-12);
        assertEquals(3.0, leaky.activate(3));

        Net sigmoid = new SigmoidNet(SHAPE, 0.05, 32, loss, r);
        assertEquals(0.5, sigmoid.activate(0), 1e-12);
        assertTrue(sigmoid.activate(20) > 0.999 && sigmoid.activate(20) <= 1.0);
        assertTrue(sigmoid.activate(-20) < 0.001 && sigmoid.activate(-20) >= 0.0);

        Net tanh = new TanhNet(SHAPE, 0.05, 32, loss, r);
        assertEquals(0.0, tanh.activate(0));
        assertEquals(-tanh.activate(1.3), tanh.activate(-1.3), 1e-12);

        Net softsign = new SoftsignNet(SHAPE, 0.05, 32, loss, r);
        assertEquals(0.0, softsign.activate(0));
        assertEquals(0.5, softsign.activate(1), 1e-12);
        assertEquals(-0.5, softsign.activate(-1), 1e-12);
    }

    @Test
    void everyActivationLearnsALinearTarget() {
        double[][] s = linearSample(400);
        for (Net net : all(new SquaredLoss(), 0.5, 3)) {
            double before = net.train(s[0], s[1], 1);
            double after = net.train(s[0], s[1], 300);
            assertTrue(Double.isFinite(after), net.id() + " diverged");
            assertTrue(after < before / 5, net.id() + ": MSE " + before + " -> " + after);
            assertEquals(0.5, net.predict(0.0), 0.05, net.id());
        }
    }

    @Test
    void weightInitialisationDiffersBetweenSaturatingAndRectifiedUnits() {
        assertNotEquals(new ReluNet(SHAPE, 0.05, 32, new SquaredLoss(), new Random(1)).predict(0.5),
                new SigmoidNet(SHAPE, 0.05, 32, new SquaredLoss(), new Random(1)).predict(0.5));
    }

    @Test
    void idReportsActivationAndLoss() {
        Random r = new Random(1);
        assertEquals("relu/squared", new ReluNet(SHAPE, 0.05, 32, new SquaredLoss(), r).id());
        assertEquals("leaky_relu/absolute", new LeakyReluNet(SHAPE, 0.05, 32, new AbsoluteLoss(), r).id());
        assertEquals("sigmoid/huber(0.01)", new SigmoidNet(SHAPE, 0.05, 32, new HuberLoss(0.01), r).id());
        assertEquals("tanh/logcosh(0.01)", new TanhNet(SHAPE, 0.05, 32, new LogCoshLoss(0.01), r).id());
        assertEquals("softsign/power(4.0,0.1)", new SoftsignNet(SHAPE, 0.05, 32, new PowerLoss(4, 0.1), r).id());
    }

    @Test
    void everyNetIsUsableThroughTheModelInterface() {
        double[][] s = linearSample(200);
        BiFunction<Model, Double, Double> ask = Model::predict;
        for (Net net : all(new SquaredLoss(), 0.5, 9)) {
            Model model = net;
            model.train(s[0], s[1], 50);
            assertTrue(Double.isFinite(ask.apply(model, 0.3)), net.id());
        }
    }

    @Test
    void layerSizesAreReportedAsACopy() {
        Net net = new ReluNet(SHAPE, 0.05, 32, new SquaredLoss(), new Random(1));
        int[] sizes = net.layerSizes();
        assertArrayEquals(SHAPE, sizes);
        sizes[1] = 999;
        assertArrayEquals(SHAPE, net.layerSizes());
    }

    @Test
    void leakySlopeIsValidated() {
        Random r = new Random(1);
        Loss loss = new SquaredLoss();
        assertThrows(IllegalArgumentException.class, () -> new LeakyReluNet(SHAPE, 0.05, 32, loss, r, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new LeakyReluNet(SHAPE, 0.05, 32, loss, r, 1.0));
        assertDoesNotThrow(() -> new LeakyReluNet(SHAPE, 0.05, 32, loss, r, 0.2));
    }

    @Test
    void nullLossIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> new ReluNet(SHAPE, 0.05, 32, null, new Random(1)));
    }
}
