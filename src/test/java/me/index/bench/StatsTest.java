package me.index.bench;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StatsTest {
    @Test
    void basicStatistics() {
        double[] v = {4, 1, 3, 2};
        assertEquals(1, Stats.min(v));
        assertEquals(4, Stats.max(v));
        assertEquals(2.5, Stats.mean(v));
        assertEquals(Math.sqrt(5.0 / 3), Stats.stdev(v), 1e-12);
        assertEquals(0, Stats.stdev(new double[]{7}));
        assertArrayEquals(new double[]{4, 1, 3, 2}, v, "statistics must not reorder the input");
    }

    @Test
    void medianAndPercentilesInterpolate() {
        assertEquals(2, Stats.median(new double[]{3, 1, 2}));
        assertEquals(2.5, Stats.median(new double[]{4, 1, 3, 2}));
        double[] v = {10, 20, 30, 40, 50};
        assertEquals(10, Stats.percentile(v, 0));
        assertEquals(50, Stats.percentile(v, 100));
        assertEquals(46, Stats.percentile(v, 90), 1e-12);
        assertEquals(5, Stats.percentile(new double[]{5}, 99));
        assertThrows(IllegalArgumentException.class, () -> Stats.percentile(v, 101));
        assertThrows(IllegalArgumentException.class, () -> Stats.percentile(v, Double.NaN));
    }

    @Test
    void coefficientOfVariationWorksOnASubrange() {
        double[] v = {1000, 10, 10, 10, -5};
        assertEquals(0, Stats.cv(v, 1, 4));
        assertEquals(Double.POSITIVE_INFINITY, Stats.cv(v, 1, 2));
        assertEquals(Stats.stdev(new double[]{1000, 10}) / 505, Stats.cv(v, 0, 2), 1e-12);
        assertEquals(0, Stats.cv(new double[]{0, 0}, 0, 2));
        assertEquals(Double.POSITIVE_INFINITY, Stats.cv(new double[]{-1, 1}, 0, 2));
    }

    @Test
    void nonFiniteValuesCanBeFilteredAndEmptyInputIsRejected() {
        assertArrayEquals(new double[]{1, 2}, Stats.finite(new double[]{Double.NaN, 1, Double.POSITIVE_INFINITY, 2}));
        assertThrows(IllegalArgumentException.class, () -> Stats.median(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> Stats.mean(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> Stats.min(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> Stats.max(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> Stats.stdev(new double[0]));
    }
}
