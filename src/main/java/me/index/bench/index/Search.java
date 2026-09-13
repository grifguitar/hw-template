package me.index.bench.index;

public final class Search {
    private Search() {
    }

    public static int lowerBound(long[] a, int from, int to, long key) {
        int lo = from;
        int hi = to;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] < key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    public static int branchlessLowerBound(long[] a, int from, int to, long key) {
        int len = to - from;
        if (len <= 0) {
            return from;
        }
        int base = from;
        while (len > 1) {
            int half = len >>> 1;
            base = (a[base + half - 1] < key) ? base + half : base;
            len -= half;
        }
        return (a[base] < key) ? base + 1 : base;
    }
}
