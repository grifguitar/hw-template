package me.index.bench.data;

import me.index.bench.TestKeys;
import me.index.io.ReadUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BinFilesTest {
    @Test
    void keysRoundTripInTheSosdLayout(@TempDir Path dir) throws IOException {
        long[] keys = TestKeys.withExtremes(10_000, 1);
        Path file = dir.resolve("keys.bin");
        BinFiles.writeKeys(file, keys);
        assertEquals(8 + 8L * keys.length, Files.size(file));
        assertArrayEquals(keys, BinFiles.readKeys(file));
        assertArrayEquals(keys, ReadUtils.read(file, keys.length, true, false, false),
                "the SOSD reader must understand the keys file");
    }

    @Test
    void anEmptyKeysetRoundTrips(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("empty.bin");
        BinFiles.writeKeys(file, new long[0]);
        assertEquals(0, BinFiles.readKeys(file).length);
    }

    @Test
    void queriesRoundTrip(@TempDir Path dir) throws IOException {
        Queries q = new Queries(new long[]{Long.MIN_VALUE, -1, 7, Long.MAX_VALUE}, new int[]{0, 3, Integer.MAX_VALUE, 2});
        Path file = dir.resolve("q.bin");
        BinFiles.writeQueries(file, q);
        assertEquals(8 + 12L * 4, Files.size(file));
        Queries read = BinFiles.readQueries(file);
        assertArrayEquals(q.keys(), read.keys());
        assertArrayEquals(q.expected(), read.expected());
    }

    @Test
    void overwritingWithFewerRecordsShrinksTheFile(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("keys.bin");
        BinFiles.writeKeys(file, TestKeys.randomSorted(1000, 2));
        long[] small = {1, 2, 3};
        BinFiles.writeKeys(file, small);
        assertArrayEquals(small, BinFiles.readKeys(file));
    }

    @Test
    void malformedFilesAreRejected(@TempDir Path dir) throws IOException {
        assertThrows(IOException.class, () -> BinFiles.readKeys(dir.resolve("absent")));

        Path tiny = dir.resolve("tiny");
        Files.write(tiny, new byte[3]);
        assertTrue(assertThrows(IOException.class, () -> BinFiles.readKeys(tiny)).getMessage().contains("too short"));

        Path lying = dir.resolve("lying");
        Files.write(lying, ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN).putLong(5).putLong(1).array());
        assertTrue(assertThrows(IOException.class, () -> BinFiles.readKeys(lying)).getMessage().contains("declares"));
        assertThrows(IOException.class, () -> BinFiles.readQueries(lying));

        Path noQueries = dir.resolve("none");
        Files.write(noQueries, new byte[8]);
        assertThrows(IOException.class, () -> BinFiles.readQueries(noQueries));
    }
}
