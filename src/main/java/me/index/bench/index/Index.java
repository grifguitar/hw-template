package me.index.bench.index;

public interface Index {
    String name();

    /**
     * Returns the first position whose key is not less than {@code key}, or the number of keys if there is none.
     */
    int lowerBound(long key);

    /**
     * Memory the index needs on top of the sorted key array.
     */
    long sizeBytes();
}
