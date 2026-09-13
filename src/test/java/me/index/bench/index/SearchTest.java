package me.index.bench.index;

import me.index.bench.TestKeys;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SearchTest {
    private static int linear(long[] a, int from, int to, long key) {
        for (int i = from; i < to; i++) {
            if (a[i] >= key) {
                return i;
            }
        }
        return to;
    }

    @Test
    void bothSearchesMatchALinearScanOnEverySubrange() {
        long[] a = {Long.MIN_VALUE, -9, -3, 0, 4, 5, 17, 100, Long.MAX_VALUE};
        for (int from = 0; from <= a.length; from++) {
            for (int to = from; to <= a.length; to++) {
                for (long probe : TestKeys.probes(a, 20, 1)) {
                    int expected = linear(a, from, to, probe);
                    int f = from;
                    int t = to;
                    assertEquals(expected, Search.lowerBound(a, from, to, probe),
                            () -> "lowerBound [" + f + ", " + t + ") key " + probe);
                    assertEquals(expected, Search.branchlessLowerBound(a, from, to, probe),
                            () -> "branchlessLowerBound [" + f + ", " + t + ") key " + probe);
                }
            }
        }
    }

    @Test
    void bothSearchesMatchALinearScanOnRandomArraysOfEverySmallLength() {
        for (int n = 1; n <= 130; n++) {
            long[] a = TestKeys.randomSorted(n, n);
            for (long probe : TestKeys.probes(a, 50, n)) {
                int expected = linear(a, 0, n, probe);
                assertEquals(expected, Search.lowerBound(a, 0, n, probe));
                assertEquals(expected, Search.branchlessLowerBound(a, 0, n, probe));
            }
        }
    }

    @Test
    void anEmptyRangeReturnsItsStart() {
        long[] a = {1, 2, 3};
        assertEquals(2, Search.lowerBound(a, 2, 2, 100));
        assertEquals(2, Search.branchlessLowerBound(a, 2, 2, 100));
        assertEquals(0, Search.branchlessLowerBound(new long[0], 0, 0, 5));
    }
}
