package me.index.bench.data;

public enum QueryOrder {
    /** Queries in the order they were drawn. */
    random,
    /** Queries sorted by key: neighbouring lookups share cache lines. */
    sorted
}
