package me.index.bench.index;

import me.index.bench.TestKeys;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KeyNormalizerTest {
    @Test
    void theEndsOfTheRangeMapToMinusOneAndOne() {
        KeyNormalizer n = new KeyNormalizer(-1000, 3000);
        assertEquals(-1.0, n.normalize(-1000));
        assertEquals(1.0, n.normalize(3000), 1e-12);
        assertEquals(0.0, n.normalize(1000), 1e-12);
    }

    @Test
    void theFullSignedRangeKeepsItsOrderAndItsMiddle() {
        KeyNormalizer n = new KeyNormalizer(Long.MIN_VALUE, Long.MAX_VALUE);
        assertEquals(-1.0, n.normalize(Long.MIN_VALUE));
        assertEquals(0.0, n.normalize(0), 1e-12);
        assertEquals(1.0, n.normalize(Long.MAX_VALUE), 1e-12);

        long[] keys = TestKeys.withExtremes(20_000, 5);
        for (int i = 1; i < keys.length; i++) {
            assertTrue(n.normalize(keys[i - 1]) <= n.normalize(keys[i]), "order broken at " + keys[i]);
        }
    }

    @Test
    void neighbouringSmallKeysStayDistinct() {
        KeyNormalizer n = new KeyNormalizer(0, 4_000_000_000L);
        for (long k = 0; k < 10_000; k++) {
            assertTrue(n.normalize(k) < n.normalize(k + 1), "keys " + k + " and " + (k + 1) + " collapsed");
        }
    }

    @Test
    void unsignedConversionCoversTheWholeRange() {
        assertEquals(0.0, KeyNormalizer.unsignedToDouble(0));
        assertEquals(12345.0, KeyNormalizer.unsignedToDouble(12345));
        assertEquals(0x1p63, KeyNormalizer.unsignedToDouble(Long.MIN_VALUE));
        assertEquals(0x1p64, KeyNormalizer.unsignedToDouble(-1L));
        assertEquals(0x1p63, KeyNormalizer.unsignedToDouble(Long.MAX_VALUE));
        assertTrue(KeyNormalizer.unsignedToDouble(Long.MAX_VALUE) <= KeyNormalizer.unsignedToDouble(Long.MIN_VALUE));
    }

    @Test
    void anEmptyOrInvertedRangeIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new KeyNormalizer(5, 5));
        assertThrows(IllegalArgumentException.class, () -> new KeyNormalizer(6, 5));
    }
}
