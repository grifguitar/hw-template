package me.index.bench.index;

import me.index.bench.TestKeys;
import me.index.config.parameters.enums.Activation;
import me.index.config.parameters.enums.LossFunction;
import me.index.math.Utils;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NetIndexTest {
    private static final Trainer.Settings QUICK = new Trainer.Settings(3, 2000, 32, LossFunction._squared, Double.NaN, 1);

    private static List<long[]> keysets() {
        return List.of(
                Utils.genKeysGaussian(20_000, true, new Random(3)),
                Utils.genKeysLognormal(20_000, true, new Random(4)),
                Utils.genKeysGaussian(20_000, false, new Random(5)),
                TestKeys.withExtremes(5_000, 6));
    }

    static void assertLookups(Index index, long[] keys) {
        int n = keys.length;
        for (int i = 0; i < n; i++) {
            int position = i;
            assertEquals(i, index.lowerBound(keys[i]), () -> index.name() + " stored key #" + position);
            if (i + 1 < n && keys[i] + 1 < keys[i + 1]) {
                assertEquals(i + 1, index.lowerBound(keys[i] + 1), () -> index.name() + " gap after #" + position);
            }
        }
        if (keys[0] > Long.MIN_VALUE) {
            assertEquals(0, index.lowerBound(keys[0] - 1));
            assertEquals(0, index.lowerBound(Long.MIN_VALUE));
        }
        if (keys[n - 1] < Long.MAX_VALUE) {
            assertEquals(n, index.lowerBound(keys[n - 1] + 1));
            assertEquals(n, index.lowerBound(Long.MAX_VALUE));
        }
        SplittableRandom rnd = new SplittableRandom(11);
        for (int i = 0; i < 5000; i++) {
            long key = rnd.nextLong(keys[0] / 2 - 1, keys[n - 1] / 2 + 1) * 2 + rnd.nextInt(2);
            assertEquals(TestKeys.lowerBound(keys, key), index.lowerBound(key), () -> index.name() + " key " + key);
        }
    }

    @Test
    void everyStoredKeyEveryGapAndBothEndsAreFoundByEveryNet() {
        List<NetSpec> specs = new ArrayList<>();
        for (Activation activation : Activation.values()) {
            specs.add(new NetSpec(activation, List.of(4, 4)));
        }
        specs.add(NetSpec.parse("_relu:none"));
        for (long[] keys : keysets()) {
            for (NetSpec spec : specs) {
                assertLookups(Trainer.train(keys, spec, QUICK).index(keys), keys);
            }
        }
    }

    @Test
    void boundsAreTheExactExtremesOfThePredictionError() {
        long[] keys = keysets().getFirst();
        NetModel model = Trainer.train(keys, NetSpec.parse("_relu:8"), QUICK);
        NetIndex index = model.index(keys);
        int maxLo = 0;
        int maxHi = 0;
        double sum = 0;
        for (int i = 0; i < keys.length; i++) {
            int d = index.position(keys[i]) - i;
            maxLo = Math.max(maxLo, d);
            maxHi = Math.max(maxHi, -d);
            sum += Math.abs(d);
        }
        assertEquals(maxLo, model.errLo());
        assertEquals(maxHi, model.errHi());
        assertEquals(maxLo, index.errLo());
        assertEquals(maxHi, index.errHi());
        assertEquals((long) maxLo + maxHi + 1, index.window());
        assertEquals(sum / keys.length, model.meanAbsErr(), 1e-9);
    }

    @Test
    void boundsDoNotDependOnHowTheKeysAreSplitAcrossThreads() {
        long[] keys = Utils.genKeysLognormal(300_000, true, new Random(8));
        Kernel kernel = Trainer.train(keys, NetSpec.parse("_softsign:4x4"), QUICK).kernel();
        NetIndex.Bounds parallel = NetIndex.bounds(keys, kernel);
        KeyNormalizer normalizer = new KeyNormalizer(keys[0], keys[keys.length - 1]);
        int lo = 0;
        int hi = 0;
        for (int i = 0; i < keys.length; i++) {
            int d = NetIndex.position(kernel, normalizer, keys.length - 1, keys[i]) - i;
            lo = Math.max(lo, d);
            hi = Math.max(hi, -d);
        }
        assertEquals(lo, parallel.errLo());
        assertEquals(hi, parallel.errHi());
    }

    @Test
    void aDivergedNetStillAnswersEveryLookup() {
        long[] keys = TestKeys.randomSorted(3000, 2);
        double[] nan = new double[Kernel.parameterCount(new int[]{1, 4, 1})];
        Arrays.fill(nan, Double.NaN);
        Kernel kernel = Kernel.of(Activation._relu, new int[]{1, 4, 1}, nan);
        NetIndex.Bounds bounds = NetIndex.bounds(keys, kernel);
        assertEquals(0, bounds.errLo());
        assertEquals(keys.length - 1, bounds.errHi());
        assertLookups(new NetIndex("nn_nan", keys, kernel, bounds.errLo(), bounds.errHi()), keys);
    }

    @Test
    void boundsThatAreTooTightFallBackToAFullSearch() {
        long[] keys = TestKeys.randomSorted(3000, 3);
        Kernel alwaysFirst = Kernel.of(Activation._relu, new int[]{1, 1}, new double[]{0, 0});
        Kernel alwaysLast = Kernel.of(Activation._relu, new int[]{1, 1}, new double[]{0, keys.length - 1});
        Kernel middle = Kernel.of(Activation._relu, new int[]{1, 1}, new double[]{0, keys.length / 2.0});
        for (Kernel kernel : List.of(alwaysFirst, alwaysLast, middle)) {
            assertLookups(new NetIndex("nn_wrong", keys, kernel, 0, 0), keys);
            assertLookups(new NetIndex("nn_wrong", keys, kernel, 3, 1), keys);
        }
    }

    @Test
    void predictedPositionsAreClampedToTheKeyArray() {
        long[] keys = {10, 20, 30, 40};
        assertEquals(3, new NetIndex("hi", keys, Kernel.of(Activation._relu, new int[]{1, 1}, new double[]{0, 1e30}), 0, 0).position(25));
        assertEquals(0, new NetIndex("lo", keys, Kernel.of(Activation._relu, new int[]{1, 1}, new double[]{0, -1e30}), 0, 0).position(25));
        assertEquals(0, new NetIndex("nan", keys, Kernel.of(Activation._relu, new int[]{1, 1}, new double[]{0, Double.NaN}), 0, 0).position(25));
        assertEquals(2, new NetIndex("two", keys, Kernel.of(Activation._relu, new int[]{1, 1}, new double[]{0, 2.4}), 0, 0).position(25));
    }

    @Test
    void theIndexKeepsItsOwnKernelCopyAndReportsItsSize() {
        long[] keys = {1, 2, 3, 4, 5};
        Kernel kernel = Kernel.of(Activation._relu, new int[]{1, 3, 1}, new double[10]);
        NetIndex index = new NetIndex("nn_relu_3", keys, kernel, 4, 4);
        assertEquals("nn_relu_3", index.name());
        assertEquals(kernel.sizeBytes(), index.sizeBytes());
        assertTrue(index.sizeBytes() > 0);
    }

    @Test
    void invalidConstructionIsRejected() {
        Kernel kernel = Kernel.of(Activation._relu, new int[]{1, 1}, new double[2]);
        assertThrows(IllegalArgumentException.class, () -> new NetIndex("x", new long[]{1}, kernel, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new NetIndex("x", new long[]{2, 1}, kernel, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> new NetIndex("x", new long[]{1, 2}, kernel, -1, 0));
        assertThrows(IllegalArgumentException.class, () -> new NetIndex("x", new long[]{1, 2}, kernel, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> NetIndex.bounds(new long[]{1}, kernel));
    }
}
