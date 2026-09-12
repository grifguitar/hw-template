package me.index.math;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;

public final class Maths {
    private Maths() {
    }

    public static final Int128 ZERO = new Int128(0, 0);
    public static final Int128 ONE = new Int128(0, 1);

    public static final Int128 TWO64 = new Int128(1, 0);
    public static final Int128 NEG_TWO64 = new Int128(-1, 0);
    public static final Int128 U64_MAX = new Int128(0, -1);

    public static final Frac128 F_ZERO = new Frac128(ZERO, ONE);
    public static final Frac128 INF_NEG = new Frac128(NEG_TWO64, ONE);
    public static final Frac128 INF_POS = new Frac128(TWO64, ONE);

    public static boolean lessZero(Int128 o) {
        return o.hi() < 0;
    }

    public static boolean eqZero(Int128 o) {
        return o.hi() == 0 && o.lo() == 0;
    }

    public static boolean greatZero(Int128 o) {
        return o.hi() > 0 || o.hi() == 0 && o.lo() != 0;
    }

    public static boolean greatOrEqZero(Int128 o) {
        return o.hi() >= 0;
    }

    public static boolean lessOrEqZero(Int128 o) {
        return o.hi() < 0 || o.hi() == 0 && o.lo() == 0;
    }

    public static boolean less(Frac128 x, Frac128 y) {
        return lessZero(diff(x, y));
    }

    public static boolean lessEq(Frac128 x, Frac128 y) {
        return lessOrEqZero(diff(x, y));
    }

    public static boolean great(Frac128 x, Frac128 y) {
        return greatZero(diff(x, y));
    }

    public static boolean greatEq(Frac128 x, Frac128 y) {
        return greatOrEqZero(diff(x, y));
    }

    public static boolean eq(Frac128 x, Frac128 y) {
        return eqZero(diff(x, y));
    }

    public static Int128 diff(Frac128 x, Frac128 y) {
        return sub(mul(x.num(), y.den()), mul(y.num(), x.den()));
    }

    public static Int128 neg(Int128 o) {
        long lo = ~o.lo() + 1;
        long hi = ~o.hi() + (lo == 0 ? 1 : 0);
        return new Int128(hi, lo);
    }

    public static Int128 sum(Int128 x, Int128 y) {
        long lo = x.lo() + y.lo();
        long hi = x.hi() + y.hi() + (Long.compareUnsigned(lo, x.lo()) < 0 ? 1L : 0L);
        return new Int128(hi, lo);
    }

    public static Int128 sub(Int128 x, Int128 y) {
        return new Int128(
                x.hi() - y.hi() - ((Long.compareUnsigned(x.lo(), y.lo()) < 0) ? 1L : 0L),
                x.lo() - y.lo());
    }

    public static Int128 sub(long x, long y) {
        return new Int128(
                ((x < 0) ? -1L : 0L) - ((y < 0) ? -1L : 0L) - ((Long.compareUnsigned(x, y) < 0) ? 1L : 0L),
                x - y);
    }

    public static Int128 mul(Int128 x, long y) {
        boolean sign = lessZero(x) ^ (y < 0);
        if (lessZero(x))
            x = neg(x);
        if (y < 0) {
            y = -y;
        }
        Int128 r = new Int128(Math.unsignedMultiplyHigh(x.lo(), y) + x.hi() * y, x.lo() * y);
        return (sign) ? neg(r) : r;
    }

    public static Int128 mul(Int128 x, Int128 y) {
        boolean sign = lessZero(x) ^ lessZero(y);
        if (lessZero(x))
            x = neg(x);
        if (lessZero(y))
            y = neg(y);
        Int128 r = new Int128(
                Math.unsignedMultiplyHigh(x.lo(), y.lo()) + x.hi() * y.lo() + x.lo() * y.hi(),
                x.lo() * y.lo());
        return (sign) ? neg(r) : r;
    }

    public static byte[] toByteArray(Int128 first) {
        return new byte[]{
                (byte) (first.hi() >>> 56),
                (byte) (first.hi() >>> 48),
                (byte) (first.hi() >>> 40),
                (byte) (first.hi() >>> 32),
                (byte) (first.hi() >>> 24),
                (byte) (first.hi() >>> 16),
                (byte) (first.hi() >>> 8),
                (byte) first.hi(),
                (byte) (first.lo() >>> 56),
                (byte) (first.lo() >>> 48),
                (byte) (first.lo() >>> 40),
                (byte) (first.lo() >>> 32),
                (byte) (first.lo() >>> 24),
                (byte) (first.lo() >>> 16),
                (byte) (first.lo() >>> 8),
                (byte) first.lo()
        };
    }

    public static BigDecimal toBigDecimal(Int128 first) {
        return new BigDecimal(new BigInteger(toByteArray(first)));
    }

    public static double toDouble(Frac128 first) {
        return (toBigDecimal(first.num()).divide(toBigDecimal(first.den()), MathContext.DECIMAL128)).doubleValue();
    }

    public static int predict(LRM lrm, long key) {
        return Math.max((int) (lrm.k() * key + lrm.b()), 0);
    }

    public static int predict(double k, double b, long key) {
        return Math.max((int) (k * key + b), 0);
    }

    public static int predict(double[] cf, long key) {
        return Math.max((int) (cf[0] * key + cf[1]), 0);
    }
}
