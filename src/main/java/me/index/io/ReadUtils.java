package me.index.io;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public final class ReadUtils {
    private ReadUtils() {
    }

    private static long readLong(DataInputStream dis, boolean isLong) throws IOException {
        long result = 0; // little endian
        for (int i = 0; i < (isLong ? 8 : 4); i++) {
            result |= ((long) dis.readUnsignedByte()) << (8 * i);
        }
        return result;
    }

    public static long[] read(String name, int maxSize, boolean isL, boolean shift, boolean plus) {
        try (DataInputStream stream = new DataInputStream(new BufferedInputStream(new FileInputStream(name)))) {
            int n = (int) readLong(stream, true);
            long[] arr = new long[Math.min(n, maxSize)];
            int i = 0;
            int j = 0;
            while (i < Math.min(n, maxSize) && j < n) {
                long x = (shift)
                        ? readLong(stream, isL) + Long.MIN_VALUE + (plus ? 1L : 0L)
                        : readLong(stream, isL);
                if (i == 0 || arr[i - 1] != x) arr[i++] = x;
                j++;
            }
            return Arrays.copyOf(arr, i);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
