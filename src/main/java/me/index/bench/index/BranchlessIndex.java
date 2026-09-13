package me.index.bench.index;

import me.index.bench.data.Keys;

public final class BranchlessIndex implements Index {
    public static final String NAME = "branchless";

    private final long[] keys;

    public BranchlessIndex(long[] keys) {
        Keys.requireStrictlyIncreasing(keys);
        this.keys = keys;
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public int lowerBound(long key) {
        return Search.branchlessLowerBound(keys, 0, keys.length, key);
    }

    @Override
    public long sizeBytes() {
        return 0;
    }
}
