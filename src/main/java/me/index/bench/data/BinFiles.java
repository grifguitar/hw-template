package me.index.bench.data;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Memory-mapped binary files, little-endian.
 * Keys: {@code uint64 count, count x int64} (the SOSD layout). Queries: {@code uint64 count, count x int64 key,
 * count x int32 expected position}. Reading copies straight into the target arrays, leaving no heap garbage.
 */
public final class BinFiles {
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final long HEADER = Long.BYTES;

    private BinFiles() {
    }

    public static void writeKeys(Path file, long[] keys) throws IOException {
        long size = HEADER + (long) Long.BYTES * keys.length;
        try (FileChannel channel = openForWriting(file); Arena arena = Arena.ofConfined()) {
            MemorySegment out = channel.map(FileChannel.MapMode.READ_WRITE, 0, size, arena);
            out.set(LONG_LE, 0, keys.length);
            MemorySegment.copy(keys, 0, out, LONG_LE, HEADER, keys.length);
        }
    }

    public static long[] readKeys(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ); Arena arena = Arena.ofConfined()) {
            long size = channel.size();
            MemorySegment in = map(file, channel, size, arena);
            int count = count(file, in, size, Long.BYTES);
            long[] keys = new long[count];
            MemorySegment.copy(in, LONG_LE, HEADER, keys, 0, count);
            return keys;
        }
    }

    public static void writeQueries(Path file, Queries queries) throws IOException {
        int count = queries.size();
        long size = HEADER + (long) (Long.BYTES + Integer.BYTES) * count;
        try (FileChannel channel = openForWriting(file); Arena arena = Arena.ofConfined()) {
            MemorySegment out = channel.map(FileChannel.MapMode.READ_WRITE, 0, size, arena);
            out.set(LONG_LE, 0, count);
            MemorySegment.copy(queries.keys(), 0, out, LONG_LE, HEADER, count);
            MemorySegment.copy(queries.expected(), 0, out, INT_LE, HEADER + (long) Long.BYTES * count, count);
        }
    }

    public static Queries readQueries(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ); Arena arena = Arena.ofConfined()) {
            long size = channel.size();
            MemorySegment in = map(file, channel, size, arena);
            int count = count(file, in, size, Long.BYTES + Integer.BYTES);
            long[] keys = new long[count];
            int[] expected = new int[count];
            MemorySegment.copy(in, LONG_LE, HEADER, keys, 0, count);
            MemorySegment.copy(in, INT_LE, HEADER + (long) Long.BYTES * count, expected, 0, count);
            try {
                return new Queries(keys, expected);
            } catch (IllegalArgumentException e) {
                throw new IOException("queries file " + file.toAbsolutePath() + " is invalid: " + e.getMessage(), e);
            }
        }
    }

    private static FileChannel openForWriting(Path file) throws IOException {
        return FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.READ, StandardOpenOption.WRITE);
    }

    private static MemorySegment map(Path file, FileChannel channel, long size, Arena arena) throws IOException {
        if (size < HEADER) {
            throw new IOException(file.toAbsolutePath() + " is too short to hold a header (" + size + " bytes)");
        }
        return channel.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
    }

    private static int count(Path file, MemorySegment in, long size, int recordBytes) throws IOException {
        long count = in.get(LONG_LE, 0);
        if (count < 0 || count > Integer.MAX_VALUE - 8 || size != HEADER + count * recordBytes) {
            throw new IOException(file.toAbsolutePath() + " declares " + Long.toUnsignedString(count)
                    + " records of " + recordBytes + " bytes but has " + size + " bytes");
        }
        return (int) count;
    }
}
