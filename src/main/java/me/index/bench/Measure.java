package me.index.bench;

import me.index.bench.data.Queries;
import me.index.bench.index.Index;
import me.index.bench.index.NetIndex;

import java.lang.management.CompilationMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

/**
 * Timing loops. A pass walks all queries in blocks: the hot work sits in {@link #lookupBlock}, which is called
 * thousands of times per pass and therefore gets a regular (not on-stack-replacement) JIT compilation,
 * and a {@code System.nanoTime()} pair per block gives the tail-latency distribution at negligible cost.
 */
public final class Measure {
    static final int STABLE_WINDOW = 5;

    private static final ThreadMXBean THREADS = ManagementFactory.getThreadMXBean();
    private static final CompilationMXBean COMPILER = ManagementFactory.getCompilationMXBean();

    private Measure() {
    }

    /**
     * @param warmupCv warm-up ends once the last {@value #STABLE_WINDOW} passes vary less than this
     *                 (but never before {@code warmupMin} passes and never after {@code warmupMax})
     */
    public record Settings(int runs, int warmupMin, int warmupMax, double warmupCv, int blockSize) {
        public Settings {
            if (runs < 1) {
                throw new IllegalArgumentException("runs must be positive, got " + runs);
            }
            if (warmupMin < 0 || warmupMax < warmupMin) {
                throw new IllegalArgumentException("warm-up passes must satisfy 0 <= min <= max, got "
                        + warmupMin + " and " + warmupMax);
            }
            if (!(warmupCv >= 0) || !Double.isFinite(warmupCv)) {
                throw new IllegalArgumentException("warm-up CV must be finite and non-negative, got " + warmupCv);
            }
            if (blockSize < 1) {
                throw new IllegalArgumentException("block size must be positive, got " + blockSize);
            }
        }
    }

    /**
     * @param allocatedBytes heap allocated by the measuring thread during the measured passes, -1 if unknown
     * @param jitMillis      JIT compilation time spent during the measured passes, -1 if unknown
     */
    public record Result(double[] runNsPerQuery, double[] blockNsPerQuery, int warmupPasses,
                         long allocatedBytes, long jitMillis, long checksum) {
    }

    public static void verify(Index index, Queries queries) {
        long[] q = queries.keys();
        int[] expected = queries.expected();
        for (int i = 0; i < q.length; i++) {
            int got = index.lowerBound(q[i]);
            if (got != expected[i]) {
                throw new IllegalStateException(index.name() + " returned " + got + " for query #" + i
                        + " (key " + q[i] + "), expected " + expected[i]);
            }
        }
    }

    public static Result lookups(Index index, Queries queries, Settings settings) {
        verify(index, queries);
        long[] q = queries.keys();
        int m = q.length;
        long expected = queries.checksum();
        int block = settings.blockSize();
        int blocks = (m + block - 1) / block;

        long[] warmNanos = new long[blocks];
        double[] warm = new double[Math.max(1, settings.warmupMax())];
        int warmup = 0;
        while (warmup < settings.warmupMax()) {
            requireChecksum(index, timedLookups(index, q, block, warmNanos, 0), expected);
            warm[warmup++] = sum(warmNanos, 0, blocks) / (double) m;
            if (warmup >= Math.max(settings.warmupMin(), STABLE_WINDOW)
                    && Stats.cv(warm, warmup - STABLE_WINDOW, warmup) <= settings.warmupCv()) {
                break;
            }
        }

        int runs = settings.runs();
        long[] nanos = new long[Math.multiplyExact(runs, blocks)];
        long[] sums = new long[runs];
        allocatedBytes();
        jitMillis();
        long jitBefore = jitMillis();
        long allocBefore = allocatedBytes();
        for (int r = 0; r < runs; r++) {
            sums[r] = timedLookups(index, q, block, nanos, r * blocks);
        }
        long allocAfter = allocatedBytes();
        long jitAfter = jitMillis();

        for (long s : sums) {
            requireChecksum(index, s, expected);
        }
        double[] runNs = new double[runs];
        double[] blockNs = new double[runs * blocks];
        for (int r = 0; r < runs; r++) {
            runNs[r] = sum(nanos, r * blocks, blocks) / (double) m;
            for (int b = 0; b < blocks; b++) {
                int len = Math.min(block, m - b * block);
                blockNs[r * blocks + b] = nanos[r * blocks + b] / (double) len;
            }
        }
        return new Result(runNs, blockNs, warmup,
                allocBefore < 0 || allocAfter < 0 ? -1 : allocAfter - allocBefore,
                jitBefore < 0 || jitAfter < 0 ? -1 : jitAfter - jitBefore,
                expected);
    }

    /**
     * Median time of the model alone (key normalisation, forward pass, rounding), without the last-mile search.
     */
    public static double modelNsPerQuery(NetIndex index, Queries queries, Settings settings) {
        long[] q = queries.keys();
        int block = settings.blockSize();
        int warmup = Math.max(settings.warmupMin(), Math.min(settings.warmupMax(), STABLE_WINDOW));
        long reference = 0;
        for (int i = 0; i < warmup; i++) {
            reference = timedPositions(index, q, block)[1];
        }
        double[] ns = new double[settings.runs()];
        for (int r = 0; r < ns.length; r++) {
            long[] timed = timedPositions(index, q, block);
            if (warmup > 0 && timed[1] != reference) {
                throw new IllegalStateException(index.name() + " predicted different positions across passes");
            }
            reference = timed[1];
            ns[r] = timed[0] / (double) q.length;
        }
        return Stats.median(ns);
    }

    static long timedLookups(Index index, long[] q, int block, long[] nanos, int base) {
        long sum = 0;
        int m = q.length;
        int b = base;
        long t0 = System.nanoTime();
        for (int from = 0; from < m; from += block) {
            sum += lookupBlock(index, q, from, Math.min(m, from + block));
            long t1 = System.nanoTime();
            nanos[b++] = t1 - t0;
            t0 = t1;
        }
        return sum;
    }

    static long lookupBlock(Index index, long[] q, int from, int to) {
        long sum = 0;
        for (int i = from; i < to; i++) {
            sum += index.lowerBound(q[i]);
        }
        return sum;
    }

    private static long[] timedPositions(NetIndex index, long[] q, int block) {
        long sum = 0;
        int m = q.length;
        long start = System.nanoTime();
        for (int from = 0; from < m; from += block) {
            sum += positionBlock(index, q, from, Math.min(m, from + block));
        }
        return new long[]{System.nanoTime() - start, sum};
    }

    static long positionBlock(NetIndex index, long[] q, int from, int to) {
        long sum = 0;
        for (int i = from; i < to; i++) {
            sum += index.position(q[i]);
        }
        return sum;
    }

    static long allocatedBytes() {
        if (THREADS instanceof com.sun.management.ThreadMXBean threads
                && threads.isThreadAllocatedMemorySupported() && threads.isThreadAllocatedMemoryEnabled()) {
            return threads.getCurrentThreadAllocatedBytes();
        }
        return -1;
    }

    static long jitMillis() {
        return COMPILER != null && COMPILER.isCompilationTimeMonitoringSupported()
                ? COMPILER.getTotalCompilationTime() : -1;
    }

    private static void requireChecksum(Index index, long actual, long expected) {
        if (actual != expected) {
            throw new IllegalStateException(index.name() + " returned positions summing to " + actual
                    + " during a timed pass, expected " + expected);
        }
    }

    private static long sum(long[] values, int from, int count) {
        long s = 0;
        for (int i = from; i < from + count; i++) {
            s += values[i];
        }
        return s;
    }
}
