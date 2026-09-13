package me.index.bench;

import me.index.bench.data.Queries;
import me.index.bench.data.QueryGenerator;
import me.index.bench.data.QueryOrder;
import me.index.bench.index.BinarySearchIndex;
import me.index.bench.index.BranchlessIndex;
import me.index.bench.index.Index;
import me.index.bench.index.NetIndex;
import me.index.bench.index.NetSpec;
import me.index.bench.index.Trainer;
import me.index.config.parameters.enums.LossFunction;
import me.index.config.parameters.enums.Workload;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MeasureTest {
    private static final long[] KEYS = TestKeys.randomSorted(5000, 1);

    private static Queries queries(int count) {
        return QueryGenerator.generate(KEYS, Workload._uniform, QueryOrder.random, count, 7);
    }

    private static NetIndex net() {
        return Trainer.train(KEYS, NetSpec.parse("_relu:4x4"),
                new Trainer.Settings(2, 1000, 32, LossFunction._squared, Double.NaN, 1)).index(KEYS);
    }

    /**
     * Answers correctly for the first {@code correctCalls} lookups, then one position too far.
     */
    private static Index drifting(String name, int correctCalls) {
        Index real = new BinarySearchIndex(KEYS);
        return new Index() {
            private int calls;

            @Override
            public String name() {
                return name;
            }

            @Override
            public int lowerBound(long key) {
                return real.lowerBound(key) + (++calls > correctCalls ? 1 : 0);
            }

            @Override
            public long sizeBytes() {
                return 0;
            }
        };
    }

    @Test
    void lookupsReportOneTimingPerRunAndPerBlock() {
        Queries q = queries(2500);
        Measure.Result r = Measure.lookups(new BinarySearchIndex(KEYS), q, new Measure.Settings(4, 2, 3, 0.5, 1000));
        assertEquals(4, r.runNsPerQuery().length);
        assertEquals(4 * 3, r.blockNsPerQuery().length);
        for (double ns : r.runNsPerQuery()) {
            assertTrue(Double.isFinite(ns) && ns >= 0, "run " + ns);
        }
        for (double ns : r.blockNsPerQuery()) {
            assertTrue(Double.isFinite(ns) && ns >= 0, "block " + ns);
        }
        assertEquals(q.checksum(), r.checksum());
        assertTrue(r.warmupPasses() >= 2 && r.warmupPasses() <= 3, "warm-up passes " + r.warmupPasses());
    }

    @Test
    void theTimedPassesDoNotAllocate() {
        Queries q = queries(20_000);
        for (Index index : List.of(new BinarySearchIndex(KEYS), new BranchlessIndex(KEYS), net())) {
            Measure.Result r = Measure.lookups(index, q, new Measure.Settings(3, 3, 3, 0, 256));
            if (r.allocatedBytes() >= 0) {
                assertEquals(0, r.allocatedBytes(), index.name() + " allocated while being timed");
            }
        }
    }

    @Test
    void aWrongAnswerFailsVerification() {
        Index wrong = drifting("off-by-one", 0);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> Measure.lookups(wrong, queries(100), new Measure.Settings(1, 0, 0, 0, 10)));
        assertTrue(e.getMessage().contains("off-by-one") && e.getMessage().contains("query #0"), e.getMessage());
    }

    @Test
    void anAnswerThatChangesAfterVerificationIsCaughtByTheChecksum() {
        Queries q = queries(1000);
        IllegalStateException inWarmup = assertThrows(IllegalStateException.class,
                () -> Measure.lookups(drifting("flaky", 1000), q, new Measure.Settings(1, 0, 2, 0, 100)));
        assertTrue(inWarmup.getMessage().contains("flaky") && inWarmup.getMessage().contains("summing"),
                inWarmup.getMessage());

        IllegalStateException inRuns = assertThrows(IllegalStateException.class,
                () -> Measure.lookups(drifting("late", 1000), q, new Measure.Settings(1, 0, 0, 0, 100)));
        assertTrue(inRuns.getMessage().contains("late"), inRuns.getMessage());
    }

    @Test
    void warmUpStopsOnceStableButRespectsItsLimits() {
        Index index = new BinarySearchIndex(KEYS);
        Queries q = queries(500);
        assertEquals(Measure.STABLE_WINDOW, Measure.lookups(index, q, new Measure.Settings(1, 0, 20, 1e9, 100)).warmupPasses());
        assertEquals(7, Measure.lookups(index, q, new Measure.Settings(1, 7, 20, 1e9, 100)).warmupPasses());
        assertEquals(3, Measure.lookups(index, q, new Measure.Settings(1, 0, 3, 1e9, 100)).warmupPasses());
        assertEquals(0, Measure.lookups(index, q, new Measure.Settings(1, 0, 0, 1e9, 100)).warmupPasses());
    }

    @Test
    void modelTimeIsMeasuredForNets() {
        double ns = Measure.modelNsPerQuery(net(), queries(5000), new Measure.Settings(3, 1, 2, 0.1, 500));
        assertTrue(Double.isFinite(ns) && ns >= 0, "model ns " + ns);
    }

    @Test
    void invalidSettingsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Measure.Settings(0, 0, 0, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Measure.Settings(1, -1, 0, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Measure.Settings(1, 5, 4, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new Measure.Settings(1, 0, 0, -0.1, 1));
        assertThrows(IllegalArgumentException.class, () -> new Measure.Settings(1, 0, 0, Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> new Measure.Settings(1, 0, 0, 0, 0));
    }
}
