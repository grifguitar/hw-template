package me.index.bench;

import java.util.Arrays;

public final class Stats {
    private Stats() {
    }

    public static double min(double[] values) {
        requireNonEmpty(values);
        double min = values[0];
        for (double v : values) {
            min = Math.min(min, v);
        }
        return min;
    }

    public static double max(double[] values) {
        requireNonEmpty(values);
        double max = values[0];
        for (double v : values) {
            max = Math.max(max, v);
        }
        return max;
    }

    public static double mean(double[] values) {
        requireNonEmpty(values);
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    /**
     * Sample standard deviation; 0 for a single value.
     */
    public static double stdev(double[] values) {
        requireNonEmpty(values);
        if (values.length < 2) {
            return 0;
        }
        double mean = mean(values);
        double sum = 0;
        for (double v : values) {
            sum += (v - mean) * (v - mean);
        }
        return Math.sqrt(sum / (values.length - 1));
    }

    public static double median(double[] values) {
        return percentile(values, 50);
    }

    /**
     * Percentile with linear interpolation between the closest ranks.
     */
    public static double percentile(double[] values, double p) {
        requireNonEmpty(values);
        if (!(p >= 0 && p <= 100)) {
            throw new IllegalArgumentException("percentile must lie in [0, 100], got " + p);
        }
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        double rank = p / 100 * (sorted.length - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
    }

    /**
     * Coefficient of variation of {@code values[from, to)}; allocation-free, so it can run between timed passes.
     */
    public static double cv(double[] values, int from, int to) {
        int n = to - from;
        if (n < 2) {
            return Double.POSITIVE_INFINITY;
        }
        double sum = 0;
        for (int i = from; i < to; i++) {
            sum += values[i];
        }
        double mean = sum / n;
        double sq = 0;
        for (int i = from; i < to; i++) {
            sq += (values[i] - mean) * (values[i] - mean);
        }
        double sd = Math.sqrt(sq / (n - 1));
        if (mean == 0) {
            return sd == 0 ? 0 : Double.POSITIVE_INFINITY;
        }
        return sd / Math.abs(mean);
    }

    public static double[] finite(double[] values) {
        return Arrays.stream(values).filter(Double::isFinite).toArray();
    }

    private static void requireNonEmpty(double[] values) {
        if (values.length == 0) {
            throw new IllegalArgumentException("no values");
        }
    }
}
