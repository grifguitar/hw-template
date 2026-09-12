package me.index.config.parameters.enums;

import me.index.ml.Loss;
import me.index.ml.Net;
import me.index.ml.loss.AbsoluteLoss;
import me.index.ml.loss.SquaredLoss;
import me.index.ml.models.LeakyReluNet;
import me.index.ml.models.ReluNet;
import me.index.ml.models.SigmoidNet;
import me.index.ml.models.SoftsignNet;
import me.index.ml.models.TanhNet;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ActivationWiringTest {
    private static final int[] SHAPE = {1, 4, 4, 1};

    private static final Map<Activation, Class<? extends Net>> EXPECTED = Map.of(
            Activation._relu, ReluNet.class,
            Activation._leaky_relu, LeakyReluNet.class,
            Activation._sigmoid, SigmoidNet.class,
            Activation._tanh, TanhNet.class,
            Activation._softsign, SoftsignNet.class);

    private static Net create(Activation activation, int[] shape, Loss loss) {
        return activation.create(shape, 0.05, 32, loss, new Random(1));
    }

    @Test
    void everyActivationCreatesItsOwnNetType() {
        for (Activation activation : Activation.values()) {
            Class<? extends Net> expected = EXPECTED.get(activation);
            assertNotNull(expected, activation + " is not listed in this test — add its net type");
            assertEquals(expected, create(activation, SHAPE, new SquaredLoss()).getClass(), activation.name());
        }
    }

    @Test
    void noTwoActivationsShareANetType() {
        assertEquals(Activation.values().length, EXPECTED.size(),
                "every activation must appear exactly once");
        assertEquals(Activation.values().length, Set.copyOf(EXPECTED.values()).size(),
                "two activations are wired to the same net type");
    }

    @Test
    void theCreatedNetReportsTheActivationItWasAskedFor() {
        for (Activation activation : Activation.values()) {
            String name = activation.name().substring(1);
            assertEquals(name + "/squared", create(activation, SHAPE, new SquaredLoss()).id(), activation.name());
        }
    }

    @Test
    void theShapeAndLossAreHandedToTheNet() {
        int[] shape = {1, 3, 7, 1};
        for (Activation activation : Activation.values()) {
            Net net = create(activation, shape, new AbsoluteLoss());
            assertArrayEquals(shape, net.layerSizes(), activation.name());
            assertTrue(net.id().endsWith("/absolute"), activation + " ignored the loss: " + net.id());
        }
    }

    @Test
    void everyActivationProducesAUsableNet() {
        for (Activation activation : Activation.values()) {
            Net net = create(activation, SHAPE, new SquaredLoss());
            assertTrue(Double.isFinite(net.predict(0.5)), activation.name());
        }
    }
}
