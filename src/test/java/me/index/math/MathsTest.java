package me.index.math;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;
import java.util.Random;

import static me.index.math.Maths.*;
import static org.junit.jupiter.api.Assertions.*;

class MathsTest {
    private static BigInteger big(Int128 v) {
        return new BigInteger(toByteArray(v));
    }

    private static Int128 of(BigInteger v) {
        BigInteger mod = v.mod(BigInteger.ONE.shiftLeft(128));
        return new Int128(mod.shiftRight(64).longValue(), mod.longValue());
    }

    @Test
    void comparisonsTreatLowWordAsUnsigned() {
        assertTrue(greatZero(U64_MAX));
        assertTrue(greatOrEqZero(U64_MAX));
        assertFalse(lessOrEqZero(U64_MAX));
        assertFalse(lessZero(U64_MAX));
        assertFalse(eqZero(U64_MAX));
    }

    @Test
    void comparisonsAgreeWithBigIntegerOnEdgeCases() {
        Int128[] samples = {
                ZERO, ONE, neg(ONE), TWO64, NEG_TWO64, U64_MAX, neg(U64_MAX),
                new Int128(0, Long.MIN_VALUE), new Int128(-1, 0), new Int128(-1, -1),
                new Int128(Long.MAX_VALUE, -1), new Int128(Long.MIN_VALUE, 0)
        };
        for (Int128 v : samples) {
            int sign = big(v).signum();
            assertEquals(sign < 0, lessZero(v), () -> "lessZero " + big(v));
            assertEquals(sign == 0, eqZero(v), () -> "eqZero " + big(v));
            assertEquals(sign > 0, greatZero(v), () -> "greatZero " + big(v));
            assertEquals(sign >= 0, greatOrEqZero(v), () -> "greatOrEqZero " + big(v));
            assertEquals(sign <= 0, lessOrEqZero(v), () -> "lessOrEqZero " + big(v));
        }
    }

    @Test
    void namedConstantsMatchTheirNames() {
        assertEquals(BigInteger.ONE.shiftLeft(64), big(TWO64));
        assertEquals(BigInteger.ONE.shiftLeft(64).negate(), big(NEG_TWO64));
        assertEquals(BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE), big(U64_MAX));
    }

    @Test
    void mulKeepsTheHighWord() {
        assertEquals(BigInteger.ONE.shiftLeft(65), big(mul(TWO64, 2L)));
        assertEquals(BigInteger.ONE.shiftLeft(128 - 1).negate(),
                big(mul(new Int128(1, 0), -(1L << 63))));
    }

    @Test
    void mulAgreesWithBigIntegerOnRandomInputs() {
        Random rnd = new Random(7);
        BigInteger mod = BigInteger.ONE.shiftLeft(128);
        for (int i = 0; i < 5000; i++) {
            Int128 x = of(new BigInteger(rnd.nextInt(60) + 1, rnd).multiply(rnd.nextBoolean()
                    ? BigInteger.ONE : BigInteger.valueOf(-1)));
            Int128 y = of(new BigInteger(rnd.nextInt(60) + 1, rnd).multiply(rnd.nextBoolean()
                    ? BigInteger.ONE : BigInteger.valueOf(-1)));
            assertEquals(big(x).multiply(big(y)).mod(mod), big(mul(x, y)).mod(mod),
                    () -> big(x) + " * " + big(y));
            long s = rnd.nextLong();
            assertEquals(big(x).multiply(BigInteger.valueOf(s)).mod(mod), big(mul(x, s)).mod(mod),
                    () -> big(x) + " * " + s);
        }
    }

    @Test
    void sumSubAndNegAgreeWithBigInteger() {
        Random rnd = new Random(11);
        BigInteger mod = BigInteger.ONE.shiftLeft(128);
        for (int i = 0; i < 2000; i++) {
            Int128 x = new Int128(rnd.nextLong(), rnd.nextLong());
            Int128 y = new Int128(rnd.nextLong(), rnd.nextLong());
            assertEquals(big(x).add(big(y)).mod(mod), big(sum(x, y)).mod(mod));
            assertEquals(big(x).subtract(big(y)).mod(mod), big(sub(x, y)).mod(mod));
            assertEquals(big(x).negate().mod(mod), big(neg(x)).mod(mod));
            long p = rnd.nextLong(), q = rnd.nextLong();
            assertEquals(BigInteger.valueOf(p).subtract(BigInteger.valueOf(q)), big(sub(p, q)));
        }
    }

    @Test
    void fractionComparisonsWork() {
        Frac128 half = new Frac128(ONE, new Int128(0, 2));
        Frac128 third = new Frac128(ONE, new Int128(0, 3));
        assertTrue(great(half, third));
        assertTrue(less(third, half));
        assertTrue(eq(half, new Frac128(new Int128(0, 2), new Int128(0, 4))));
        assertTrue(less(INF_NEG, F_ZERO));
        assertTrue(great(INF_POS, F_ZERO));
        assertTrue(less(INF_NEG, INF_POS));
        assertTrue(lessEq(F_ZERO, F_ZERO));
        assertTrue(greatEq(F_ZERO, F_ZERO));
    }

    @Test
    void linearPredictorClampsAtZeroAndAgreesAcrossOverloads() {
        LRM model = new LRM(2.0, 3.0, 7);
        assertEquals(23, predict(model, 10L));
        assertEquals(23, predict(2.0, 3.0, 10L));
        assertEquals(23, predict(new double[]{2.0, 3.0}, 10L));
        assertEquals(0, predict(model, -100L), "a negative position must be clamped to 0");
        assertEquals(0, predict(-1.0, 0.0, 5L));
        assertEquals(0, predict(new double[]{-1.0, 0.0}, 5L));
    }
}
