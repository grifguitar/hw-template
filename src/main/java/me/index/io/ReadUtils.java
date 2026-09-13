package me.index.io;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

public final class ReadUtils {
    private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private ReadUtils() {
    }

    public static long[] read(Path file, int maxSize, boolean eightBytes, boolean shift, boolean plusOne)
            throws IOException {
        if (maxSize <= 0) {
            throw new IllegalArgumentException("maxSize must be positive, got " + maxSize);
        }
        if (!Files.isReadable(file)) {
            throw new IOException("keyset file is missing or not readable: " + file.toAbsolutePath()
                    + " (SOSD datasets are not bundled with the repo; pass their directory as the second argument)");
        }
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
             Arena arena = Arena.ofConfined()) {
            long size = channel.size();
            if (size < Long.BYTES) {
                throw truncated(file);
            }
            MemorySegment data = channel.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
            long declared = data.get(LONG_LE, 0);
            if (declared < 0 || declared > Integer.MAX_VALUE) {
                throw new IOException("keyset file " + file.toAbsolutePath()
                        + " declares an unsupported element count: " + Long.toUnsignedString(declared));
            }
            int total = (int) declared;
            int wanted = Math.min(total, maxSize);
            int width = eightBytes ? Long.BYTES : Integer.BYTES;
            long available = (size - Long.BYTES) / width;

            long[] keys = new long[wanted];
            int kept = 0;
            long offset = Long.BYTES;
            for (int read = 0; kept < wanted && read < total; read++, offset += width) {
                if (read >= available) {
                    throw truncated(file);
                }
                long key = eightBytes
                        ? data.get(LONG_LE, offset)
                        : Integer.toUnsignedLong(data.get(INT_LE, offset));
                if (shift) {
                    key += Long.MIN_VALUE + (plusOne ? 1L : 0L);
                }
                if (kept == 0 || keys[kept - 1] != key) {
                    keys[kept++] = key;
                }
            }
            return (kept == keys.length) ? keys : Arrays.copyOf(keys, kept);
        }
    }

    private static IOException truncated(Path file) {
        return new IOException("keyset file " + file.toAbsolutePath() + " ended earlier than its header promised");
    }
}
