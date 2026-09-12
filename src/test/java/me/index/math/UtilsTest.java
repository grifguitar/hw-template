package me.index.math;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class UtilsTest {
    private static void assertStrictlyIncreasing(long[] keys) {
        assertTrue(keys.length > 0, "keyset must not be empty");
        for (int i = 1; i < keys.length; i++) {
            assertTrue(keys[i - 1] < keys[i],
                    "keys must be sorted and distinct, but keys[" + (i - 1) + "] = " + keys[i - 1]
                            + " >= keys[" + i + "] = " + keys[i]);
        }
    }

    @Test
    void gaussianKeysAreSortedDistinctAndWithinRange() {
        long[] keys32 = Utils.genKeysGaussian(50_000, false, new Random(1));
        assertStrictlyIncreasing(keys32);
        assertTrue(keys32.length <= 50_000);
        for (long k : keys32) {
            assertTrue(Math.abs(k) <= Integer.MAX_VALUE, "32-bit keyset leaked out of range: " + k);
        }

        long[] keys64 = Utils.genKeysGaussian(50_000, true, new Random(1));
        assertStrictlyIncreasing(keys64);
    }

    @Test
    void lognormalKeysAreSortedDistinctAndPositive() {
        long[] keys = Utils.genKeysLognormal(50_000, true, new Random(1));
        assertStrictlyIncreasing(keys);
        assertTrue(keys[0] >= 1, "lognormal keys must be positive, got " + keys[0]);
    }

    @Test
    void generationIsReproducibleForTheSameSeed() {
        assertArrayEquals(Utils.genKeysGaussian(10_000, true, new Random(9)),
                Utils.genKeysGaussian(10_000, true, new Random(9)));
        assertArrayEquals(Utils.genKeysLognormal(10_000, true, new Random(9)),
                Utils.genKeysLognormal(10_000, true, new Random(9)));
    }

    @Test
    void nonPositiveSizeYieldsAnEmptyArray() {
        assertEquals(0, Utils.genKeysGaussian(0, true, new Random(1)).length);
        assertEquals(0, Utils.genKeysLognormal(-5, true, new Random(1)).length);
    }
}
