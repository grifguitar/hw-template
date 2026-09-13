package me.index.bench.index;

import me.index.bench.data.Keys;

import java.util.Arrays;

public final class BinarySearchIndex implements Index {
    public static final String NAME = "binary";

    private final long[] keys;

    public BinarySearchIndex(long[] keys) {
        Keys.requireStrictlyIncreasing(keys);
        this.keys = keys;
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public int lowerBound(long key) {
        int r = Arrays.binarySearch(keys, key);
        return r >= 0 ? r : -r - 1;
    }

    @Override
    public long sizeBytes() {
        return 0;
    }
}
