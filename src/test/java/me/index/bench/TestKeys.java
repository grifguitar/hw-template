package me.index.bench;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.stream.LongStream;

public final class TestKeys {
    private TestKeys() {
    }

    public static long[] randomSorted(int n, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Set<Long> distinct = new HashSet<>();
        while (distinct.size() < n) {
            distinct.add(rnd.nextLong());
        }
        long[] keys = distinct.stream().mapToLong(Long::longValue).toArray();
        Arrays.sort(keys);
        return keys;
    }

    public static long[] withExtremes(int n, long seed) {
        long[] keys = randomSorted(n, seed);
        keys[0] = Long.MIN_VALUE;
        keys[n - 1] = Long.MAX_VALUE;
        return keys;
    }

    public static int linearLowerBound(long[] keys, long key) {
        for (int i = 0; i < keys.length; i++) {
            if (keys[i] >= key) {
                return i;
            }
        }
        return keys.length;
    }

    public static int lowerBound(long[] keys, long key) {
        int r = Arrays.binarySearch(keys, key);
        return r >= 0 ? r : -r - 1;
    }

    /**
     * Every key, both neighbours of every key, the extremes and {@code random} random longs.
     */
    public static long[] probes(long[] keys, int random, long seed) {
        LongStream.Builder b = LongStream.builder();
        b.add(Long.MIN_VALUE).add(Long.MAX_VALUE).add(0);
        for (long k : keys) {
            b.add(k);
            if (k > Long.MIN_VALUE) {
                b.add(k - 1);
            }
            if (k < Long.MAX_VALUE) {
                b.add(k + 1);
            }
        }
        SplittableRandom rnd = new SplittableRandom(seed);
        for (int i = 0; i < random; i++) {
            b.add(rnd.nextLong());
        }
        return b.build().toArray();
    }
}
