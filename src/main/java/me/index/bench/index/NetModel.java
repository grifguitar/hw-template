package me.index.bench.index;

import me.index.bench.data.Keys;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * A trained net frozen for lookups: position-scaled parameters, the keyset it belongs to and its error bounds.
 */
public record NetModel(NetSpec spec, double[] parameters, int n, long minKey, long maxKey, long keysChecksum,
                       int errLo, int errHi, double meanAbsErr,
                       String loss, int epochs, int trainSample, double trainMse, double trainSeconds) {
    private static final int MAGIC = 0x4E4E4958;
    private static final int VERSION = 1;

    public NetModel {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(loss, "loss");
        parameters = parameters.clone();
        int expected = Kernel.parameterCount(spec.layerSizes());
        if (parameters.length != expected) {
            throw new IllegalArgumentException(spec + " needs " + expected + " parameters, got " + parameters.length);
        }
        if (n < 2) {
            throw new IllegalArgumentException("a model needs at least 2 keys, got " + n);
        }
        if (minKey >= maxKey) {
            throw new IllegalArgumentException("minKey must be less than maxKey, got " + minKey + " and " + maxKey);
        }
        if (errLo < 0 || errHi < 0 || errLo >= n || errHi >= n) {
            throw new IllegalArgumentException("error bounds must lie in [0, " + (n - 1) + "], got "
                    + errLo + " and " + errHi);
        }
        if (epochs < 0 || trainSample < 0) {
            throw new IllegalArgumentException("epochs and trainSample must not be negative");
        }
    }

    @Override
    public double[] parameters() {
        return parameters.clone();
    }

    public long window() {
        return (long) errLo + errHi + 1;
    }

    public Kernel kernel() {
        return Kernel.of(spec.activation(), spec.layerSizes(), parameters);
    }

    public NetIndex index(long[] keys) {
        if (keys.length != n || keys[0] != minKey || keys[n - 1] != maxKey || Keys.checksum(keys) != keysChecksum) {
            throw new IllegalArgumentException("model " + spec.label() + " was trained on a different keyset"
                    + " (model: n=" + n + ", keys " + minKey + ".." + maxKey + "; given: n=" + keys.length + ")");
        }
        return new NetIndex(spec.label(), keys, kernel(), errLo, errHi);
    }

    public void write(Path file) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeUTF(spec.toString());
            out.writeInt(parameters.length);
            for (double p : parameters) {
                out.writeDouble(p);
            }
            out.writeInt(n);
            out.writeLong(minKey);
            out.writeLong(maxKey);
            out.writeLong(keysChecksum);
            out.writeInt(errLo);
            out.writeInt(errHi);
            out.writeDouble(meanAbsErr);
            out.writeUTF(loss);
            out.writeInt(epochs);
            out.writeInt(trainSample);
            out.writeDouble(trainMse);
            out.writeDouble(trainSeconds);
        }
    }

    public static NetModel read(Path file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            if (in.readInt() != MAGIC) {
                throw new IOException(file.toAbsolutePath() + " is not a model file");
            }
            int version = in.readInt();
            if (version != VERSION) {
                throw new IOException(file.toAbsolutePath() + " has model format version " + version
                        + ", expected " + VERSION);
            }
            NetSpec spec = NetSpec.parse(in.readUTF());
            int count = in.readInt();
            if (count != Kernel.parameterCount(spec.layerSizes())) {
                throw new IOException(file.toAbsolutePath() + " stores " + count + " parameters for " + spec);
            }
            double[] parameters = new double[count];
            for (int i = 0; i < count; i++) {
                parameters[i] = in.readDouble();
            }
            NetModel model = new NetModel(spec, parameters, in.readInt(), in.readLong(), in.readLong(), in.readLong(),
                    in.readInt(), in.readInt(), in.readDouble(),
                    in.readUTF(), in.readInt(), in.readInt(), in.readDouble(), in.readDouble());
            if (in.read() != -1) {
                throw new IOException(file.toAbsolutePath() + " has trailing bytes after the model");
            }
            return model;
        } catch (EOFException e) {
            throw new IOException("model file " + file.toAbsolutePath() + " is truncated", e);
        } catch (IllegalArgumentException e) {
            throw new IOException("model file " + file.toAbsolutePath() + " is corrupt: " + e.getMessage(), e);
        }
    }
}
