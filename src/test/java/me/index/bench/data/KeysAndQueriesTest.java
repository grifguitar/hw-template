package me.index.bench.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class KeysAndQueriesTest {
    @Test
    void strictlyIncreasingKeysAreRequired() {
        assertDoesNotThrow(() -> Keys.requireStrictlyIncreasing(new long[]{Long.MIN_VALUE, 0, Long.MAX_VALUE}));
        assertDoesNotThrow(() -> Keys.requireStrictlyIncreasing(new long[]{7}));
        assertThrows(IllegalArgumentException.class, () -> Keys.requireStrictlyIncreasing(new long[0]));
        assertThrows(IllegalArgumentException.class, () -> Keys.requireStrictlyIncreasing(new long[]{1, 1}));
        assertThrows(IllegalArgumentException.class, () -> Keys.requireStrictlyIncreasing(new long[]{2, 1}));
    }

    @Test
    void theChecksumSeesValuesOrderAndLength() {
        long base = Keys.checksum(new long[]{1, 2, 3});
        assertEquals(base, Keys.checksum(new long[]{1, 2, 3}));
        assertNotEquals(base, Keys.checksum(new long[]{1, 2, 4}));
        assertNotEquals(base, Keys.checksum(new long[]{2, 1, 3}));
        assertNotEquals(base, Keys.checksum(new long[]{1, 2, 3, 0}));
        assertNotEquals(Keys.checksum(new long[0]), Keys.checksum(new long[]{0}));
    }

    @Test
    void queriesValidateTheirShapeAndSumTheirPositions() {
        Queries q = new Queries(new long[]{10, 30, 20}, new int[]{0, 2, 1});
        assertEquals(3, q.size());
        assertEquals(3, q.checksum());
        assertThrows(IllegalArgumentException.class, () -> new Queries(new long[1], new int[2]));
        assertThrows(IllegalArgumentException.class, () -> new Queries(new long[0], new int[0]));
        assertThrows(NullPointerException.class, () -> new Queries(null, new int[0]));
        assertEquals((long) Integer.MAX_VALUE * 2, new Queries(new long[2], new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE}).checksum());
    }

    @Test
    void queriesMustDescribeLowerBoundsOfTheKeyset() {
        long[] keys = {10, 20, 30};
        assertDoesNotThrow(() -> new Queries(new long[]{10, 30, 15, 31, 5}, new int[]{0, 2, 1, 3, 0})
                .requireConsistentWith(keys));
        assertThrows(IllegalArgumentException.class, () -> new Queries(new long[]{20}, new int[]{0}).requireConsistentWith(keys));
        assertThrows(IllegalArgumentException.class, () -> new Queries(new long[]{20}, new int[]{2}).requireConsistentWith(keys));
        assertThrows(IllegalArgumentException.class, () -> new Queries(new long[]{20}, new int[]{4}).requireConsistentWith(keys));
        assertThrows(IllegalArgumentException.class, () -> new Queries(new long[]{20}, new int[]{-1}).requireConsistentWith(keys));
    }
}
