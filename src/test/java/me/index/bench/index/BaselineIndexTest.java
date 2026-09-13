package me.index.bench.index;

import me.index.bench.TestKeys;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BaselineIndexTest {
    private static List<Index> indexes(long[] keys) {
        return List.of(new BinarySearchIndex(keys), new BranchlessIndex(keys));
    }

    @Test
    void hitsMissesAndOutOfRangeKeysMatchALinearScan() {
        for (long[] keys : List.of(TestKeys.randomSorted(3000, 1), TestKeys.withExtremes(500, 2), new long[]{42},
                new long[]{-5, 5})) {
            for (Index index : indexes(keys)) {
                for (long probe : TestKeys.probes(keys, 2000, 3)) {
                    assertEquals(TestKeys.linearLowerBound(keys, probe), index.lowerBound(probe),
                            () -> index.name() + " key " + probe);
                }
            }
        }
    }

    @Test
    void baselinesHaveStableNamesAndNeedNoExtraMemory() {
        long[] keys = {1, 2, 3};
        assertEquals("binary", new BinarySearchIndex(keys).name());
        assertEquals("branchless", new BranchlessIndex(keys).name());
        for (Index index : indexes(keys)) {
            assertEquals(0, index.sizeBytes(), index.name());
        }
    }

    @Test
    void unsortedDuplicateOrEmptyKeysAreRejected() {
        for (long[] bad : List.of(new long[0], new long[]{3, 2}, new long[]{1, 1})) {
            assertThrows(IllegalArgumentException.class, () -> new BinarySearchIndex(bad));
            assertThrows(IllegalArgumentException.class, () -> new BranchlessIndex(bad));
        }
    }
}
