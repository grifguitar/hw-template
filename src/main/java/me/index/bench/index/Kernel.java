package me.index.bench.index;

import me.index.config.parameters.enums.Activation;
import me.index.ml.models.LeakyReluNet;

import java.util.Arrays;

/**
 * Inference-only copy of a {@link me.index.ml.Net}: all parameters live in one flat array
 * ({@code [w0, b0, w1, b1, ...]}, weights row-major) and the forward pass reuses two preallocated buffers.
 * The summation order matches {@code Net}, so predictions are bit-for-bit identical.
 * One subclass per activation keeps the {@link #activate} call monomorphic in a JVM that uses a single net type.
 */
public abstract class Kernel {
    private final int[] sz;
    private final double[] params;
    private final double[] bufA;
    private final double[] bufB;

    private Kernel(int[] layerSizes, double[] parameters) {
        if (layerSizes.length < 2 || layerSizes[0] != 1 || layerSizes[layerSizes.length - 1] != 1) {
            throw new IllegalArgumentException("layer sizes must start and end with 1, got "
                    + Arrays.toString(layerSizes));
        }
        int width = 1;
        for (int s : layerSizes) {
            if (s <= 0) {
                throw new IllegalArgumentException("layer sizes must be positive, got " + Arrays.toString(layerSizes));
            }
            width = Math.max(width, s);
        }
        int expected = parameterCount(layerSizes);
        if (parameters.length != expected) {
            throw new IllegalArgumentException("layers " + Arrays.toString(layerSizes) + " need " + expected
                    + " parameters, got " + parameters.length);
        }
        this.sz = layerSizes.clone();
        this.params = parameters.clone();
        this.bufA = new double[width];
        this.bufB = new double[width];
    }

    public static int parameterCount(int[] layerSizes) {
        long count = 0;
        for (int l = 0; l + 1 < layerSizes.length; l++) {
            count += (long) layerSizes[l] * layerSizes[l + 1] + layerSizes[l + 1];
        }
        if (count > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("too many parameters: " + count);
        }
        return (int) count;
    }

    public static Kernel of(Activation activation, int[] layerSizes, double[] parameters) {
        return switch (activation) {
            case _relu -> new Relu(layerSizes, parameters);
            case _leaky_relu -> new LeakyRelu(layerSizes, parameters);
            case _sigmoid -> new Sigmoid(layerSizes, parameters);
            case _tanh -> new Tanh(layerSizes, parameters);
            case _softsign -> new Softsign(layerSizes, parameters);
        };
    }

    public abstract Activation activation();

    protected abstract double activate(double z);

    public final double predict(double x) {
        double[] p = params;
        int[] sizes = sz;
        double[] cur = bufA;
        double[] next = bufB;
        cur[0] = x;
        int off = 0;
        int last = sizes.length - 2;
        for (int l = 0; l <= last; l++) {
            int in = sizes[l];
            int out = sizes[l + 1];
            int bias = off + in * out;
            for (int i = 0; i < out; i++) {
                int row = off + i * in;
                double acc = 0.0;
                for (int j = 0; j < in; j++) {
                    acc += p[row + j] * cur[j];
                }
                double v = acc + p[bias + i];
                next[i] = l < last ? activate(v) : v;
            }
            off = bias + out;
            double[] t = cur;
            cur = next;
            next = t;
        }
        return cur[0];
    }

    /**
     * Returns a kernel whose output layer is multiplied by {@code factor}.
     */
    public final Kernel scaleOutput(double factor) {
        double[] scaled = params.clone();
        int lastIn = sz[sz.length - 2];
        for (int i = scaled.length - lastIn - 1; i < scaled.length; i++) {
            scaled[i] *= factor;
        }
        return of(activation(), sz, scaled);
    }

    public final Kernel copy() {
        return of(activation(), sz, params);
    }

    public final int[] layerSizes() {
        return sz.clone();
    }

    public final double[] parameters() {
        return params.clone();
    }

    public final long sizeBytes() {
        return 8L * (params.length + bufA.length + bufB.length) + 4L * sz.length;
    }

    static final class Relu extends Kernel {
        Relu(int[] layerSizes, double[] parameters) {
            super(layerSizes, parameters);
        }

        @Override
        public Activation activation() {
            return Activation._relu;
        }

        @Override
        protected double activate(double z) {
            return Math.max(0.0, z);
        }
    }

    static final class LeakyRelu extends Kernel {
        LeakyRelu(int[] layerSizes, double[] parameters) {
            super(layerSizes, parameters);
        }

        @Override
        public Activation activation() {
            return Activation._leaky_relu;
        }

        @Override
        protected double activate(double z) {
            return z > 0 ? z : LeakyReluNet.DEFAULT_SLOPE * z;
        }
    }

    static final class Sigmoid extends Kernel {
        Sigmoid(int[] layerSizes, double[] parameters) {
            super(layerSizes, parameters);
        }

        @Override
        public Activation activation() {
            return Activation._sigmoid;
        }

        @Override
        protected double activate(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
    }

    static final class Tanh extends Kernel {
        Tanh(int[] layerSizes, double[] parameters) {
            super(layerSizes, parameters);
        }

        @Override
        public Activation activation() {
            return Activation._tanh;
        }

        @Override
        protected double activate(double z) {
            return Math.tanh(z);
        }
    }

    static final class Softsign extends Kernel {
        Softsign(int[] layerSizes, double[] parameters) {
            super(layerSizes, parameters);
        }

        @Override
        public Activation activation() {
            return Activation._softsign;
        }

        @Override
        protected double activate(double z) {
            return z / (1.0 + Math.abs(z));
        }
    }
}
