package me.index.bench.index;

import me.index.bench.data.Keys;

import java.util.stream.IntStream;

/**
 * Learned index: the kernel predicts a position, a bounded binary search finishes the lookup inside
 * {@code [pos - errLo, pos + errHi]}. The bounds are exact for every stored key; keys that are not stored
 * (for which a non-monotone net gives no guarantee) fall back to a search of the remaining range.
 */
public final class NetIndex implements Index {
    private static final int MIN_CHUNK = 1 << 16;

    public record Bounds(int errLo, int errHi, double meanAbsErr) {
        public long window() {
            return (long) errLo + errHi + 1;
        }
    }

    private final String name;
    private final long[] keys;
    private final int n;
    private final long minKey;
    private final long maxKey;
    private final KeyNormalizer normalizer;
    private final Kernel kernel;
    private final int errLo;
    private final int errHi;

    /**
     * @param positionKernel a kernel whose output is already scaled to positions ({@code 0 .. n-1})
     */
    public NetIndex(String name, long[] keys, Kernel positionKernel, int errLo, int errHi) {
        if (keys.length < 2) {
            throw new IllegalArgumentException("a learned index needs at least 2 keys, got " + keys.length);
        }
        Keys.requireStrictlyIncreasing(keys);
        if (errLo < 0 || errHi < 0 || errLo >= keys.length || errHi >= keys.length) {
            throw new IllegalArgumentException("error bounds must lie in [0, " + (keys.length - 1) + "], got "
                    + errLo + " and " + errHi);
        }
        this.name = name;
        this.keys = keys;
        this.n = keys.length;
        this.minKey = keys[0];
        this.maxKey = keys[n - 1];
        this.normalizer = new KeyNormalizer(minKey, maxKey);
        this.kernel = positionKernel.copy();
        this.errLo = errLo;
        this.errHi = errHi;
    }

    public static Bounds bounds(long[] keys, Kernel positionKernel) {
        int n = keys.length;
        if (n < 2) {
            throw new IllegalArgumentException("a learned index needs at least 2 keys, got " + n);
        }
        KeyNormalizer normalizer = new KeyNormalizer(keys[0], keys[n - 1]);
        int chunks = Math.max(1, Math.min(Runtime.getRuntime().availableProcessors(), n / MIN_CHUNK));
        int[] lo = new int[chunks];
        int[] hi = new int[chunks];
        long[] abs = new long[chunks];
        IntStream.range(0, chunks).parallel().forEach(c -> {
            Kernel k = positionKernel.copy();
            int from = (int) ((long) n * c / chunks);
            int to = (int) ((long) n * (c + 1) / chunks);
            int maxLo = 0;
            int maxHi = 0;
            long sum = 0;
            for (int i = from; i < to; i++) {
                int d = position(k, normalizer, n - 1, keys[i]) - i;
                if (d > maxLo) {
                    maxLo = d;
                } else if (-d > maxHi) {
                    maxHi = -d;
                }
                sum += Math.abs(d);
            }
            lo[c] = maxLo;
            hi[c] = maxHi;
            abs[c] = sum;
        });
        int errLo = 0;
        int errHi = 0;
        long sum = 0;
        for (int c = 0; c < chunks; c++) {
            errLo = Math.max(errLo, lo[c]);
            errHi = Math.max(errHi, hi[c]);
            sum += abs[c];
        }
        return new Bounds(errLo, errHi, (double) sum / n);
    }

    static int position(Kernel kernel, KeyNormalizer normalizer, int maxPosition, long key) {
        long p = Math.round(kernel.predict(normalizer.normalize(key)));
        return (int) Math.max(0L, Math.min(maxPosition, p));
    }

    public int position(long key) {
        return position(kernel, normalizer, n - 1, key);
    }

    @Override
    public int lowerBound(long key) {
        if (key <= minKey) {
            return 0;
        }
        if (key > maxKey) {
            return n;
        }
        int pos = position(kernel, normalizer, n - 1, key);
        int lo = Math.max(0, pos - errLo);
        int hi = (int) Math.min(n, (long) pos + errHi + 1);
        int r = Search.lowerBound(keys, lo, hi, key);
        if (r == lo && lo > 0 && keys[lo - 1] >= key) {
            return Search.lowerBound(keys, 0, lo, key);
        }
        if (r == hi && hi < n) {
            return Search.lowerBound(keys, hi, n, key);
        }
        return r;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public long sizeBytes() {
        return kernel.sizeBytes();
    }

    public int errLo() {
        return errLo;
    }

    public int errHi() {
        return errHi;
    }

    public long window() {
        return (long) errLo + errHi + 1;
    }
}
