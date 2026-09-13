package me.index.bench.data;

import java.util.Objects;

/**
 * Lookup keys with the position {@code lowerBound} must return for each of them.
 */
public record Queries(long[] keys, int[] expected) {
    public Queries {
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(expected, "expected");
        if (keys.length != expected.length) {
            throw new IllegalArgumentException("keys and expected positions differ in length: "
                    + keys.length + " vs " + expected.length);
        }
        if (keys.length == 0) {
            throw new IllegalArgumentException("queries must not be empty");
        }
    }

    public int size() {
        return keys.length;
    }

    public long checksum() {
        long sum = 0;
        for (int e : expected) {
            sum += e;
        }
        return sum;
    }

    public void requireConsistentWith(long[] sortedKeys) {
        for (int i = 0; i < keys.length; i++) {
            int p = expected[i];
            boolean ok = p >= 0 && p <= sortedKeys.length
                    && (p == sortedKeys.length || sortedKeys[p] >= keys[i])
                    && (p == 0 || sortedKeys[p - 1] < keys[i]);
            if (!ok) {
                throw new IllegalArgumentException("query #" + i + " (key " + keys[i] + ", expected position " + p
                        + ") does not belong to this keyset of " + sortedKeys.length + " keys");
            }
        }
    }
}
