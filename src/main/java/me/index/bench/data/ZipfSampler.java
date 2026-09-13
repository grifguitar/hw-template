package me.index.bench.data;

import java.util.SplittableRandom;

/**
 * Zipf distribution over ranks {@code 0 .. n-1} (rank 0 is the most popular), sampled in O(1) by
 * rejection-inversion (W. Hörmann, G. Derflinger, "Rejection-inversion to generate variates from monotone
 * discrete distributions", 1996), so no table of size {@code n} is needed.
 */
final class ZipfSampler {
    private final int n;
    private final double exponent;
    private final double hIntegralX1;
    private final double hIntegralN;
    private final double s;

    ZipfSampler(int n, double exponent) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive, got " + n);
        }
        if (!(exponent > 0) || !Double.isFinite(exponent)) {
            throw new IllegalArgumentException("exponent must be finite and positive, got " + exponent);
        }
        this.n = n;
        this.exponent = exponent;
        this.hIntegralX1 = hIntegral(1.5) - 1.0;
        this.hIntegralN = hIntegral(n + 0.5);
        this.s = 2.0 - hIntegralInverse(hIntegral(2.5) - h(2));
    }

    int sample(SplittableRandom rnd) {
        while (true) {
            double u = hIntegralN + rnd.nextDouble() * (hIntegralX1 - hIntegralN);
            double x = hIntegralInverse(u);
            int k = (int) (x + 0.5);
            if (k < 1) {
                k = 1;
            } else if (k > n) {
                k = n;
            }
            if (k - x <= s || u >= hIntegral(k + 0.5) - h(k)) {
                return k - 1;
            }
        }
    }

    private double hIntegral(double x) {
        double logX = Math.log(x);
        return helper2((1.0 - exponent) * logX) * logX;
    }

    private double h(double x) {
        return Math.exp(-exponent * Math.log(x));
    }

    private double hIntegralInverse(double x) {
        double t = x * (1.0 - exponent);
        if (t < -1.0) {
            t = -1.0;
        }
        return Math.exp(helper1(t) * x);
    }

    /** {@code log(1 + x) / x}, stable near zero. */
    private static double helper1(double x) {
        return Math.abs(x) > 1e-8 ? Math.log1p(x) / x : 1.0 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x));
    }

    /** {@code (exp(x) - 1) / x}, stable near zero. */
    private static double helper2(double x) {
        return Math.abs(x) > 1e-8 ? Math.expm1(x) / x : 1.0 + x * 0.5 * (1.0 + x / 3.0 * (1.0 + 0.25 * x));
    }
}
