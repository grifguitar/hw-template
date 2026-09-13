package me.index.bench.data;

import java.util.SplittableRandom;

/**
 * Bijection {@code rank -> (rank * a + b) mod n} that scatters popular ranks over the whole key array
 * without materialising an {@code int[n]} permutation. The multiplier sits near {@code n / golden ratio}
 * (Fibonacci hashing), so consecutive ranks land far apart; the seed shifts it slightly and picks the offset.
 */
final class AffinePermutation {
    private static final double INVERSE_GOLDEN_RATIO = 0.6180339887498949;

    private final long n;
    private final long multiplier;
    private final long offset;

    AffinePermutation(int n, SplittableRandom rnd) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive, got " + n);
        }
        this.n = n;
        if (n == 1) {
            multiplier = 1;
            offset = 0;
            return;
        }
        long m = Math.max(1, Math.min(n - 1, (long) (n * INVERSE_GOLDEN_RATIO) + rnd.nextLong(Math.max(1, n / 1000))));
        while (gcd(m, n) != 1) {
            m = m % (n - 1) + 1;
        }
        multiplier = m;
        offset = rnd.nextLong(n);
    }

    int apply(int rank) {
        return (int) ((rank * multiplier + offset) % n);
    }

    private static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }
}
