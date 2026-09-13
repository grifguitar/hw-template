package me.index.bench.index;

/**
 * Maps keys of {@code [minKey, maxKey]} onto {@code [-1, 1]}. The distance to {@code minKey} is taken in unsigned
 * 64-bit arithmetic, so keysets spanning more than {@code Long.MAX_VALUE} keep their order.
 */
public final class KeyNormalizer {
    private final long minKey;
    private final double scale;

    public KeyNormalizer(long minKey, long maxKey) {
        if (minKey >= maxKey) {
            throw new IllegalArgumentException("minKey must be less than maxKey, got " + minKey + " and " + maxKey);
        }
        this.minKey = minKey;
        this.scale = 2.0 / unsignedToDouble(maxKey - minKey);
    }

    public double normalize(long key) {
        return unsignedToDouble(key - minKey) * scale - 1.0;
    }

    public static double unsignedToDouble(long v) {
        return v >= 0 ? (double) v : (double) (v >>> 1) * 2.0;
    }
}
