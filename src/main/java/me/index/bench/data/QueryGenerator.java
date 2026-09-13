package me.index.bench.data;

import me.index.config.parameters.enums.Workload;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Draws lookups of stored keys.
 * <ul>
 *     <li>{@code _uniform}: every key equally likely;</li>
 *     <li>{@code _zipf}: Zipf with exponent {@value #ZIPF_EXPONENT}, popular keys scattered over the array;</li>
 *     <li>{@code _x_y_90}, {@code _x_y_99}, {@code _x_y_9999}: 90%, 99%, 99.99% of the lookups hit a hot set of
 *     10%, 1%, 0.01% of the keys (scattered over the array), the rest are uniform over the other keys.</li>
 * </ul>
 */
public final class QueryGenerator {
    public static final double ZIPF_EXPONENT = 0.99;

    private QueryGenerator() {
    }

    public static Queries generate(long[] keys, Workload workload, QueryOrder order, int count, long seed) {
        if (keys.length == 0) {
            throw new IllegalArgumentException("cannot draw queries from an empty keyset");
        }
        if (count <= 0) {
            throw new IllegalArgumentException("query count must be positive, got " + count);
        }
        int n = keys.length;
        SplittableRandom rnd = new SplittableRandom(seed);
        AffinePermutation scatter = new AffinePermutation(n, rnd);
        int[] positions = new int[count];
        switch (workload) {
            case _uniform -> {
                for (int i = 0; i < count; i++) {
                    positions[i] = rnd.nextInt(n);
                }
            }
            case _zipf -> {
                ZipfSampler zipf = new ZipfSampler(n, ZIPF_EXPONENT);
                for (int i = 0; i < count; i++) {
                    positions[i] = scatter.apply(zipf.sample(rnd));
                }
            }
            case _x_y_90, _x_y_99, _x_y_9999 -> hotCold(positions, n, hotQueryShare(workload), scatter, rnd);
        }
        if (order == QueryOrder.sorted) {
            Arrays.sort(positions);
        }
        long[] queries = new long[count];
        for (int i = 0; i < count; i++) {
            queries[i] = keys[positions[i]];
        }
        return new Queries(queries, positions);
    }

    public static double hotQueryShare(Workload workload) {
        return switch (workload) {
            case _x_y_90 -> 0.90;
            case _x_y_99 -> 0.99;
            case _x_y_9999 -> 0.9999;
            default -> throw new IllegalArgumentException(workload + " has no hot set");
        };
    }

    public static int hotKeys(int n, double hotQueryShare) {
        return (int) Math.max(1, Math.min(n, Math.round(n * (1.0 - hotQueryShare))));
    }

    private static void hotCold(int[] positions, int n, double share, AffinePermutation scatter,
                                SplittableRandom rnd) {
        int hot = hotKeys(n, share);
        int cold = n - hot;
        for (int i = 0; i < positions.length; i++) {
            int rank = (cold == 0 || rnd.nextDouble() < share) ? rnd.nextInt(hot) : hot + rnd.nextInt(cold);
            positions[i] = scatter.apply(rank);
        }
    }
}
