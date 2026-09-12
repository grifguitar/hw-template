package me.index.io;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ReadUtilsTest {
    private static Path write(Path dir, String name, boolean eightBytes, long... values) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 + values.length * (eightBytes ? 8 : 4))
                .order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(values.length);
        for (long v : values) {
            if (eightBytes) buf.putLong(v);
            else buf.putInt((int) v);
        }
        Path file = dir.resolve(name);
        Files.write(file, buf.array());
        return file;
    }

    @Test
    void readsAllKeys(@TempDir Path dir) throws IOException {
        Path f = write(dir, "keys", true, 1, 2, 3, 10);
        assertArrayEquals(new long[]{1, 2, 3, 10}, ReadUtils.read(f, 100, true, false, false));
    }

    @Test
    void honoursMaxSize(@TempDir Path dir) throws IOException {
        Path f = write(dir, "keys", true, 1, 2, 3, 10);
        assertArrayEquals(new long[]{1, 2}, ReadUtils.read(f, 2, true, false, false));
    }

    @Test
    void dropsAdjacentDuplicatesAndKeepsFillingUpToMaxSize(@TempDir Path dir) throws IOException {
        Path f = write(dir, "keys", true, 1, 1, 1, 2, 2, 3);
        assertArrayEquals(new long[]{1, 2, 3}, ReadUtils.read(f, 100, true, false, false));
        assertArrayEquals(new long[]{1, 2}, ReadUtils.read(f, 2, true, false, false));
    }

    @Test
    void readsFourByteKeysUnsigned(@TempDir Path dir) throws IOException {
        Path f = write(dir, "keys", false, 1, 0xFFFFFFFFL);
        assertArrayEquals(new long[]{1, 0xFFFFFFFFL}, ReadUtils.read(f, 100, false, false, false));
    }

    @Test
    void appliesShiftAndPlusOne(@TempDir Path dir) throws IOException {
        Path f = write(dir, "keys", true, 0, 1);
        assertArrayEquals(new long[]{Long.MIN_VALUE, Long.MIN_VALUE + 1},
                ReadUtils.read(f, 100, true, true, false));
        assertArrayEquals(new long[]{Long.MIN_VALUE + 1, Long.MIN_VALUE + 2},
                ReadUtils.read(f, 100, true, true, true));
    }

    @Test
    void missingFileFailsWithAReadableMessage(@TempDir Path dir) {
        IOException e = assertThrows(IOException.class,
                () -> ReadUtils.read(dir.resolve("nope"), 10, true, false, false));
        assertTrue(e.getMessage().contains("nope"), e.getMessage());
    }

    @Test
    void truncatedFileFailsWithAReadableMessage(@TempDir Path dir) throws IOException {
        Path f = dir.resolve("short");
        ByteBuffer buf = ByteBuffer.allocate(8 + 4).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(100);
        buf.putInt(1);
        Files.write(f, buf.array());
        IOException e = assertThrows(IOException.class, () -> ReadUtils.read(f, 100, true, false, false));
        assertTrue(e.getMessage().contains("earlier than its header promised"), e.getMessage());
    }

    @Test
    void rejectsNonPositiveMaxSize(@TempDir Path dir) throws IOException {
        Path f = write(dir, "keys", true, 1, 2);
        assertThrows(IllegalArgumentException.class, () -> ReadUtils.read(f, 0, true, false, false));
    }
}
