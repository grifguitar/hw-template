package me.index.math;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;
import java.util.Random;

import static me.index.math.Maths.*;
import static org.junit.jupiter.api.Assertions.*;

class Frac128Test {

    private static Int128 i(long v) {
        return new Int128(v < 0 ? -1L : 0L, v);
    }

    @Test
    void zeroDenominatorIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new Frac128(ONE, ZERO));
    }

    @Test
    void negativeDenominatorIsNormalized() {
        Frac128 f = new Frac128(ONE, i(-1));
        assertEquals(i(-1), f.num());
        assertEquals(ONE, f.den());
        assertEquals(new Frac128(i(-3), i(4)), new Frac128(i(3), i(-4)));
    }

    @Test
    void comparisonsStayCorrectWhenTheDenominatorWasNegative() {
        Frac128 minusOne = new Frac128(ONE, i(-1));
        assertTrue(less(minusOne, F_ZERO), "1 / -1 is -1 and must compare below zero");
        assertTrue(great(F_ZERO, minusOne));
        assertTrue(eq(minusOne, new Frac128(i(-1), ONE)));

        Frac128 a = new Frac128(i(1), i(-2));
        Frac128 b = new Frac128(i(1), i(-3));
        assertTrue(less(a, b), "-1/2 < -1/3");
    }

    @Test
    void comparisonsMatchExactRationalOrderOnRandomFractions() {
        Random rnd = new Random(17);
        for (int t = 0; t < 20_000; t++) {
            long an = (rnd.nextLong() >> 34), ad = (rnd.nextLong() >> 34);
            long bn = (rnd.nextLong() >> 34), bd = (rnd.nextLong() >> 34);
            if (ad == 0 || bd == 0) continue;
            Frac128 x = new Frac128(i(an), i(ad));
            Frac128 y = new Frac128(i(bn), i(bd));

            BigInteger xn = BigInteger.valueOf(an), xd = BigInteger.valueOf(ad);
            if (xd.signum() < 0) {
                xn = xn.negate();
                xd = xd.negate();
            }
            BigInteger yn = BigInteger.valueOf(bn), yd = BigInteger.valueOf(bd);
            if (yd.signum() < 0) {
                yn = yn.negate();
                yd = yd.negate();
            }
            int expect = xn.multiply(yd).compareTo(yn.multiply(xd));
            assertEquals(expect < 0, less(x, y), an + "/" + ad + " vs " + bn + "/" + bd);
            assertEquals(expect > 0, great(x, y), an + "/" + ad + " vs " + bn + "/" + bd);
            assertEquals(expect == 0, eq(x, y), an + "/" + ad + " vs " + bn + "/" + bd);
        }
    }

    @Test
    void sentinelsOrderAroundZero() {
        assertTrue(less(INF_NEG, F_ZERO));
        assertTrue(great(INF_POS, F_ZERO));
        assertTrue(less(INF_NEG, INF_POS));
    }

    @Test
    void toDoubleMatchesExactDivision() {
        assertEquals(0.5, toDouble(new Frac128(ONE, i(2))));
        assertEquals(-0.25, toDouble(new Frac128(i(1), i(-4))));
        assertEquals(0.0, toDouble(F_ZERO));
        assertEquals(Math.pow(2, 64), toDouble(new Frac128(TWO64, ONE)), Math.pow(2, 64) * 1e-15);
    }

    @Test
    void toBigDecimalRoundTripsThroughTheByteRepresentation() {
        for (Int128 v : new Int128[]{ZERO, ONE, neg(ONE), TWO64, NEG_TWO64, U64_MAX,
                new Int128(Long.MIN_VALUE, 0), new Int128(Long.MAX_VALUE, -1)}) {
            assertEquals(new BigInteger(toByteArray(v)), toBigDecimal(v).toBigIntegerExact());
        }
    }

    @Test
    void unrepresentableSignNormalizationIsRejected() {
        Int128 min = new Int128(Long.MIN_VALUE, 0);
        assertThrows(IllegalArgumentException.class, () -> new Frac128(min, i(-1)));
        assertThrows(IllegalArgumentException.class, () -> new Frac128(ONE, min));
        assertDoesNotThrow(() -> new Frac128(min, ONE));
    }
}
