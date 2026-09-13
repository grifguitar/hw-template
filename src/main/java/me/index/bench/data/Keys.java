package me.index.bench.data;

public final class Keys {
    private Keys() {
    }

    public static void requireStrictlyIncreasing(long[] keys) {
        if (keys.length == 0) {
            throw new IllegalArgumentException("keys must not be empty");
        }
        for (int i = 1; i < keys.length; i++) {
            if (keys[i - 1] >= keys[i]) {
                throw new IllegalArgumentException("keys must be strictly increasing, but keys[" + (i - 1) + "] = "
                        + keys[i - 1] + " >= keys[" + i + "] = " + keys[i]);
            }
        }
    }

    public static long checksum(long[] keys) {
        long h = 0x9E3779B97F4A7C15L ^ keys.length;
        for (long k : keys) {
            h ^= k;
            h *= 0xBF58476D1CE4E5B9L;
            h ^= h >>> 31;
        }
        return h;
    }
}
