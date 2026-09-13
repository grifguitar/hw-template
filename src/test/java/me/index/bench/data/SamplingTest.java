package me.index.bench.data;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SamplingTest {
    @Test
    void zipfSamplesStayInRange() {
        SplittableRandom rnd = new SplittableRandom(1);
        for (int n : new int[]{1, 2, 3, 10, 1000, 200_000_000}) {
            ZipfSampler zipf = new ZipfSampler(n, 0.99);
            for (int i = 0; i < 20_000; i++) {
                int k = zipf.sample(rnd);
                assertTrue(k >= 0 && k < n, "rank " + k + " for n = " + n);
            }
        }
    }

    @Test
    void zipfFrequenciesMatchTheDistribution() {
        for (double exponent : new double[]{0.5, 0.99, 1.0, 1.5}) {
            int n = 50;
            int samples = 500_000;
            ZipfSampler zipf = new ZipfSampler(n, exponent);
            SplittableRandom rnd = new SplittableRandom(2);
            int[] counts = new int[n];
            for (int i = 0; i < samples; i++) {
                counts[zipf.sample(rnd)]++;
            }
            double harmonic = 0;
            for (int k = 1; k <= n; k++) {
                harmonic += Math.pow(k, -exponent);
            }
            for (int k = 1; k <= n; k++) {
                double p = Math.pow(k, -exponent) / harmonic;
                double sigma = Math.sqrt(samples * p * (1 - p));
                assertEquals(samples * p, counts[k - 1], 5 * sigma, "exponent " + exponent + ", rank " + k);
            }
        }
    }

    @Test
    void zipfRejectsInvalidParameters() {
        assertThrows(IllegalArgumentException.class, () -> new ZipfSampler(0, 1));
        assertThrows(IllegalArgumentException.class, () -> new ZipfSampler(10, 0));
        assertThrows(IllegalArgumentException.class, () -> new ZipfSampler(10, Double.NaN));
    }

    @Test
    void affinePermutationsAreBijections() {
        for (int n : new int[]{1, 2, 3, 4, 10, 97, 1000, 1024, 65_536, 99_991}) {
            for (long seed = 0; seed < 5; seed++) {
                AffinePermutation perm = new AffinePermutation(n, new SplittableRandom(seed));
                boolean[] seen = new boolean[n];
                for (int r = 0; r < n; r++) {
                    int p = perm.apply(r);
                    assertTrue(p >= 0 && p < n);
                    assertFalse(seen[p], "n = " + n + ": position " + p + " hit twice");
                    seen[p] = true;
                }
            }
        }
    }

    @Test
    void affinePermutationsDependOnTheSeedAndHandleLargeN() {
        AffinePermutation a = new AffinePermutation(1000, new SplittableRandom(1));
        AffinePermutation b = new AffinePermutation(1000, new SplittableRandom(2));
        boolean differs = false;
        for (int r = 0; r < 1000; r++) {
            differs |= a.apply(r) != b.apply(r);
        }
        assertTrue(differs);

        int n = Integer.MAX_VALUE - 1;
        AffinePermutation large = new AffinePermutation(n, new SplittableRandom(3));
        for (int r : new int[]{0, 1, n / 2, n - 1}) {
            int p = large.apply(r);
            assertTrue(p >= 0 && p < n, "position " + p);
        }
        assertThrows(IllegalArgumentException.class, () -> new AffinePermutation(0, new SplittableRandom(1)));
    }
}
