package me.index.bench.data;

import me.index.bench.TestKeys;
import me.index.config.parameters.enums.Workload;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QueryGeneratorTest {
    private static final long[] KEYS = TestKeys.randomSorted(10_000, 1);
    private static final int COUNT = 200_000;

    private static int[] sortedCounts(int[] positions, int n) {
        int[] counts = new int[n];
        for (int p : positions) {
            counts[p]++;
        }
        Arrays.sort(counts);
        return counts;
    }

    private static double topShare(int[] positions, int n, int top) {
        int[] counts = sortedCounts(positions, n);
        long sum = 0;
        for (int i = n - top; i < n; i++) {
            sum += counts[i];
        }
        return (double) sum / positions.length;
    }

    @Test
    void everyQueryIsAStoredKeyWithItsPosition() {
        for (Workload workload : Workload.values()) {
            for (QueryOrder order : QueryOrder.values()) {
                Queries q = QueryGenerator.generate(KEYS, workload, order, 5000, 3);
                assertEquals(5000, q.size());
                for (int i = 0; i < q.size(); i++) {
                    assertEquals(KEYS[q.expected()[i]], q.keys()[i], workload + " " + order + " #" + i);
                }
                q.requireConsistentWith(KEYS);
            }
        }
    }

    @Test
    void sortedOrderSortsTheSameQueriesThatRandomOrderDraws() {
        for (Workload workload : Workload.values()) {
            Queries random = QueryGenerator.generate(KEYS, workload, QueryOrder.random, 5000, 4);
            Queries sorted = QueryGenerator.generate(KEYS, workload, QueryOrder.sorted, 5000, 4);
            int[] expected = random.expected().clone();
            Arrays.sort(expected);
            assertArrayEquals(expected, sorted.expected(), workload.name());
            for (int i = 1; i < sorted.size(); i++) {
                assertTrue(sorted.keys()[i - 1] <= sorted.keys()[i]);
            }
            if (Arrays.stream(random.expected()).distinct().count() > 1) {
                assertFalse(Arrays.equals(random.expected(), sorted.expected()), workload + " random order came out sorted");
            }
        }
    }

    @Test
    void generationIsDeterministicPerSeed() {
        Queries a = QueryGenerator.generate(KEYS, Workload._zipf, QueryOrder.random, 1000, 5);
        Queries b = QueryGenerator.generate(KEYS, Workload._zipf, QueryOrder.random, 1000, 5);
        Queries c = QueryGenerator.generate(KEYS, Workload._zipf, QueryOrder.random, 1000, 6);
        assertArrayEquals(a.keys(), b.keys());
        assertArrayEquals(a.expected(), b.expected());
        assertFalse(Arrays.equals(a.expected(), c.expected()));
    }

    @Test
    void uniformQueriesSpreadEvenly() {
        int[] p = QueryGenerator.generate(KEYS, Workload._uniform, QueryOrder.random, COUNT, 7).expected();
        double mean = Arrays.stream(p).average().orElseThrow();
        assertEquals((KEYS.length - 1) / 2.0, mean, KEYS.length * 0.01);
        assertTrue(topShare(p, KEYS.length, 100) < 0.03, "uniform queries must not have a hot set");
    }

    @Test
    void zipfQueriesFollowTheZipfLawAndAreScattered() {
        int n = KEYS.length;
        int[] p = QueryGenerator.generate(KEYS, Workload._zipf, QueryOrder.random, COUNT, 8).expected();
        double harmonic = 0;
        for (int k = 1; k <= n; k++) {
            harmonic += Math.pow(k, -QueryGenerator.ZIPF_EXPONENT);
        }
        int[] counts = sortedCounts(p, n);
        assertEquals(1.0 / harmonic, (double) counts[n - 1] / COUNT, 0.1 / harmonic, "share of the most popular key");

        int[] byPosition = new int[n];
        for (int x : p) {
            byPosition[x]++;
        }
        int hottest = 0;
        for (int i = 1; i < n; i++) {
            if (byPosition[i] > byPosition[hottest]) {
                hottest = i;
            }
        }
        long firstHundred = Arrays.stream(byPosition, 0, 100).asLongStream().sum();
        assertTrue(firstHundred < COUNT / 10, "popular keys must be scattered, not packed at the start");
        assertTrue(byPosition[hottest] > COUNT / 20);
    }

    @Test
    void hotSetWorkloadsSendTheirShareToTheHotKeys() {
        record Case(Workload workload, int n, double share) {
        }
        for (Case c : new Case[]{new Case(Workload._x_y_90, 10_000, 0.90), new Case(Workload._x_y_99, 10_000, 0.99),
                new Case(Workload._x_y_9999, 100_000, 0.9999)}) {
            long[] keys = TestKeys.randomSorted(c.n(), 9);
            int hot = QueryGenerator.hotKeys(c.n(), c.share());
            assertEquals(Math.round(c.n() * (1 - c.share())), hot, c.workload().name());
            int[] p = QueryGenerator.generate(keys, c.workload(), QueryOrder.random, COUNT, 10).expected();
            double share = topShare(p, c.n(), hot);
            assertEquals(c.share(), share, 0.005, c.workload().name());

            int[] byPosition = new int[c.n()];
            for (int x : p) {
                byPosition[x]++;
            }
            int min = c.n();
            int max = -1;
            int threshold = COUNT / hot / 2;
            for (int i = 0; i < c.n(); i++) {
                if (byPosition[i] > threshold) {
                    min = Math.min(min, i);
                    max = Math.max(max, i);
                }
            }
            assertTrue(hot == 1 || max - min > c.n() / 2, c.workload() + ": hot keys must be scattered");
            assertEquals(c.share(), QueryGenerator.hotQueryShare(c.workload()));
        }
        assertThrows(IllegalArgumentException.class, () -> QueryGenerator.hotQueryShare(Workload._zipf));
    }

    @Test
    void tinyKeysetsWork() {
        for (Workload workload : Workload.values()) {
            Queries q = QueryGenerator.generate(new long[]{5}, workload, QueryOrder.random, 100, 1);
            assertTrue(Arrays.stream(q.expected()).allMatch(x -> x == 0), workload.name());
            QueryGenerator.generate(new long[]{5, 6}, workload, QueryOrder.sorted, 100, 1).requireConsistentWith(new long[]{5, 6});
        }
    }

    @Test
    void invalidArgumentsAreRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> QueryGenerator.generate(new long[0], Workload._uniform, QueryOrder.random, 10, 1));
        assertThrows(IllegalArgumentException.class,
                () -> QueryGenerator.generate(KEYS, Workload._uniform, QueryOrder.random, 0, 1));
    }
}
