package me.index.ml;

import me.index.ml.loss.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LossTest {

    private static final Loss[] ALL = {new SquaredLoss(), new AbsoluteLoss(), new HuberLoss(0.05),
            new LogCoshLoss(0.05), new PowerLoss(3, 0.1), new PowerLoss(2), new PowerLoss(4)};

    @Test
    void squaredGradientIsTheResidual() {
        Loss l = new SquaredLoss();
        assertEquals(0.25, l.gradient(0.75, 0.5), 1e-15);
        assertEquals(-0.25, l.gradient(0.25, 0.5), 1e-15);
        assertEquals(0.0, l.gradient(0.5, 0.5));
        assertEquals("squared", l.id());
    }

    @Test
    void absoluteGradientOnlyCarriesTheSign() {
        Loss l = new AbsoluteLoss();
        assertEquals(1.0, l.gradient(0.9, 0.1));
        assertEquals(1.0, l.gradient(0.11, 0.1));
        assertEquals(-1.0, l.gradient(0.0, 0.1));
        assertEquals(0.0, l.gradient(0.1, 0.1));
    }

    @Test
    void huberIsSquaredNearZeroAndAbsoluteFarFromIt() {
        Loss huber = new HuberLoss(0.01);
        Loss squared = new SquaredLoss();
        assertEquals(squared.gradient(0.505, 0.5), huber.gradient(0.505, 0.5), 1e-15);
        assertEquals(0.01, huber.gradient(0.9, 0.5), 1e-15);
        assertEquals(-0.01, huber.gradient(0.1, 0.5), 1e-15);
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss(0));
        assertThrows(IllegalArgumentException.class, () -> new HuberLoss(Double.NaN));
    }

    @Test
    void logCoshIsSmoothAndBounded() {
        Loss l = new LogCoshLoss(0.01);
        assertEquals(0.0, l.gradient(0.5, 0.5));
        assertTrue(l.gradient(0.9, 0.5) <= 0.01, "gradient must saturate at the scale");
        assertTrue(l.gradient(-0.9, 0.5) >= -0.01);
        assertEquals(0.01, l.gradient(0.9, 0.5), 1e-12);
        assertTrue(l.gradient(0.505, 0.5) < 0.01, "below the scale the gradient must stay small");
        assertEquals(new SquaredLoss().gradient(0.5001, 0.5), l.gradient(0.5001, 0.5), 1e-8);
        assertThrows(IllegalArgumentException.class, () -> new LogCoshLoss(-1));
    }

    @Test
    void powerLossGrowsFasterThanSquaredOnLargeErrors() {
        Loss squared = new SquaredLoss();
        Loss power = new PowerLoss(4, 0.1);
        double small = 0.01, large = 0.2;
        double ratioSquared = Math.abs(squared.gradient(0.5 + large, 0.5) / squared.gradient(0.5 + small, 0.5));
        double ratioPower = Math.abs(power.gradient(0.5 + large, 0.5) / power.gradient(0.5 + small, 0.5));
        assertTrue(ratioPower > ratioSquared * 10,
                "power loss must punish big errors much harder: " + ratioPower + " vs " + ratioSquared);
        assertEquals(1.0, Math.signum(power.gradient(0.9, 0.5)));
        assertEquals(-1.0, Math.signum(power.gradient(0.1, 0.5)));
    }

    @Test
    void powerLossWithExponentTwoMatchesScaledSquaredLoss() {
        Loss power = new PowerLoss(2, 1.0);
        Loss squared = new SquaredLoss();
        for (double e = -0.5; e <= 0.5; e += 0.05) {
            assertEquals(squared.gradient(0.5 + e, 0.5), power.gradient(0.5 + e, 0.5), 1e-12);
        }
    }

    @Test
    void allGradientsPointUphill() {
        Loss[] losses = {new SquaredLoss(), new AbsoluteLoss(), new HuberLoss(0.01),
                new LogCoshLoss(0.01), new PowerLoss(3, 0.03), new PowerLoss(6, 0.3)};
        for (Loss l : losses) {
            assertTrue(l.gradient(0.7, 0.5) > 0, l.id() + " must push down when predicting too high");
            assertTrue(l.gradient(0.3, 0.5) < 0, l.id() + " must push up when predicting too low");
            assertEquals(0.0, l.gradient(0.5, 0.5), 1e-300, l.id() + " must vanish at the target");
        }
    }

    @Test
    void everyGradientIsTheDerivativeOfItsValue() {
        double h = 1e-7;
        for (Loss l : ALL) {
            for (double e = -0.9; e <= 0.9; e += 0.017) {
                double numeric = (l.value(0.5 + e + h, 0.5) - l.value(0.5 + e - h, 0.5)) / (2 * h);
                assertEquals(numeric, l.gradient(0.5 + e, 0.5), 1e-5,
                        l.id() + " at residual " + e);
            }
        }
    }

    @Test
    void everyValueIsZeroAtTheTargetAndPositiveAroundIt() {
        for (Loss l : ALL) {
            assertEquals(0.0, l.value(0.5, 0.5), 1e-300, l.id());
            assertTrue(l.value(0.6, 0.5) > 0, l.id());
            assertTrue(l.value(0.4, 0.5) > 0, l.id());
            assertEquals(l.value(0.6, 0.5), l.value(0.4, 0.5), 1e-15, l.id() + " must be symmetric");
            assertTrue(l.value(0.9, 0.5) > l.value(0.6, 0.5), l.id() + " must grow with the residual");
        }
    }

    @Test
    void everyLossIsHalfSquaredNearTheTarget() {
        double e = 1e-4;
        for (Loss l : new Loss[]{new SquaredLoss(), new HuberLoss(0.05), new LogCoshLoss(0.05), new PowerLoss(2)}) {
            assertEquals(0.5 * e * e, l.value(0.5 + e, 0.5), 1e-14, l.id());
        }
    }

    @Test
    void huberValueIsContinuousAtTheTransition() {
        double delta = 0.25;
        Loss l = new HuberLoss(delta);
        assertEquals(0.5 * delta * delta, l.value(delta, 0.0), 1e-15, "quadratic branch up to delta");
        assertEquals(l.value(delta - 1e-9, 0.0), l.value(delta + 1e-9, 0.0), 1e-8, "no jump at delta");
        assertEquals(delta * (1.0 - 0.5 * delta), l.value(1.0, 0.0), 1e-15, "linear branch beyond delta");
    }

    @Test
    void logCoshValueStaysFiniteOnHugeResiduals() {
        Loss l = new LogCoshLoss(1e-3);
        double far = l.value(1e6, 0.0);
        assertTrue(Double.isFinite(far), "log cosh must not overflow: " + far);
        assertEquals(1e-3 * 1e6 - 1e-6 * Math.log(2), far, 1e-3);
    }

    @Test
    void powerLossDefaultScaleKeepsGradientsBoundedOnNormalisedTargets() {
        for (double p = 1; p <= 6; p += 0.5) {
            Loss l = new PowerLoss(p);
            for (double e = -1.0; e <= 1.0; e += 0.05) {
                assertTrue(Math.abs(l.gradient(0.5 + e, 0.5)) <= 1.0 + 1e-12,
                        l.id() + " blew up at residual " + e);
            }
        }
    }

    @Test
    void powerLossWithoutAScaleIsTheUnitLpLoss() {
        assertEquals(new SquaredLoss().value(0.8, 0.5), new PowerLoss(2).value(0.8, 0.5), 1e-15);
        assertEquals(new SquaredLoss().gradient(0.8, 0.5), new PowerLoss(2).gradient(0.8, 0.5), 1e-15);
        assertEquals(1.0, PowerLoss.DEFAULT_SCALE);
    }

    @Test
    void invalidParametersAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new PowerLoss(0.5, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new PowerLoss(4, 0));
        assertThrows(IllegalArgumentException.class, () -> new PowerLoss(Double.POSITIVE_INFINITY, 1));
    }
}
