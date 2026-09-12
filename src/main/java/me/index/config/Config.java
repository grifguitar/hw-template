package me.index.config;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

public record Config(
        Keyset keyset,
        Workload workload,
        WorkloadPerm workloadPerm,
        WorkloadFactor workloadFactor,
        DataSize dataSize,
        MaxErr maxErr,
        Training training
) {
    public static final Workload DEFAULT_WORKLOAD = Workload._uniform;
    public static final WorkloadPerm DEFAULT_WORKLOAD_PERM = WorkloadPerm._false;
    public static final WorkloadFactor DEFAULT_WORKLOAD_FACTOR = WorkloadFactor._read_only;
    public static final MaxErr DEFAULT_MAX_ERR = MaxErr._0;

    public Config {
        if (keyset == null) throw new IllegalArgumentException("keyset must not be null");
        if (workload == null) throw new IllegalArgumentException("workload must not be null");
        if (workloadPerm == null) throw new IllegalArgumentException("workloadPerm must not be null");
        if (workloadFactor == null) throw new IllegalArgumentException("workloadFactor must not be null");
        if (dataSize == null) throw new IllegalArgumentException("dataSize must not be null");
        if (maxErr == null) throw new IllegalArgumentException("maxErr must not be null");
        if (training == null) throw new IllegalArgumentException("training must not be null");
    }

    public static Config read(Properties p) {
        return new Config(
                required(p, "keyset", Keyset.class),
                optional(p, "workload.distribution", Workload.class, DEFAULT_WORKLOAD),
                optional(p, "workload.permutation", WorkloadPerm.class, DEFAULT_WORKLOAD_PERM),
                optional(p, "workload.factor", WorkloadFactor.class, DEFAULT_WORKLOAD_FACTOR),
                required(p, "data.size", DataSize.class),
                optional(p, "max.err", MaxErr.class, DEFAULT_MAX_ERR),
                readTraining(p)
        );
    }

    public static Config load(Path file) throws IOException {
        Properties props = new Properties();
        try (InputStream in = Files.newInputStream(file)) {
            props.load(in);
        } catch (IOException e) {
            throw new IOException("cannot read config file " + file.toAbsolutePath(), e);
        }
        try {
            return read(props);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("invalid config " + file.toAbsolutePath() + ": " + e.getMessage(), e);
        }
    }

    private static Training readTraining(Properties p) {
        return new Training(
                optionalLong(p, "train.seed", Training.DEFAULT_SEED),
                optionalPositiveDouble(p, "train.learning.rate", Training.DEFAULT_LEARNING_RATE),
                optionalPositiveInt(p, "train.batch.size", Training.DEFAULT_BATCH_SIZE),
                optionalPositiveInt(p, "train.epochs", Training.DEFAULT_EPOCHS),
                optionalIntList(p, "net.hidden.layers", Training.DEFAULT_HIDDEN_LAYERS),
                optionalString(p, "plot.path", Training.DEFAULT_PLOT_PATH)
        );
    }

    private static <E extends Enum<E>> E required(Properties p, String key, Class<E> type) {
        String raw = trimmed(p, key);
        if (raw == null) {
            throw new IllegalArgumentException("missing required property '" + key + "'; allowed values: "
                    + Arrays.toString(type.getEnumConstants()));
        }
        return parseEnum(key, raw, type);
    }

    private static <E extends Enum<E>> E optional(Properties p, String key, Class<E> type, E fallback) {
        String raw = trimmed(p, key);
        return raw == null ? fallback : parseEnum(key, raw, type);
    }

    private static long optionalLong(Properties p, String key, long fallback) {
        String raw = trimmed(p, key);
        if (raw == null) return fallback;
        try {
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("property '" + key + "': expected an integer, got '" + raw + "'", e);
        }
    }

    private static int optionalPositiveInt(Properties p, String key, int fallback) {
        long v = optionalLong(p, key, fallback);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("property '" + key + "': expected a positive int, got " + v);
        }
        return (int) v;
    }

    private static double optionalPositiveDouble(Properties p, String key, double fallback) {
        String raw = trimmed(p, key);
        if (raw == null) return fallback;
        double v;
        try {
            v = Double.parseDouble(raw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("property '" + key + "': expected a number, got '" + raw + "'", e);
        }
        if (!(v > 0) || !Double.isFinite(v)) {
            throw new IllegalArgumentException("property '" + key + "': expected a finite positive number, got " + v);
        }
        return v;
    }

    private static String optionalString(Properties p, String key, String fallback) {
        String raw = trimmed(p, key);
        return raw == null ? fallback : raw;
    }

    private static List<Integer> optionalIntList(Properties p, String key, List<Integer> fallback) {
        String raw = trimmed(p, key);
        if (raw == null) return fallback;
        if (raw.isEmpty()) return List.of();
        List<Integer> out = new ArrayList<>();
        for (String part : raw.split(",", -1)) {
            String s = part.trim();
            int v;
            try {
                v = Integer.parseInt(s);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "property '" + key + "': expected comma-separated ints, got '" + raw + "'", e);
            }
            if (v <= 0) {
                throw new IllegalArgumentException("property '" + key + "': sizes must be positive, got " + v);
            }
            out.add(v);
        }
        return List.copyOf(out);
    }

    private static <E extends Enum<E>> E parseEnum(String key, String raw, Class<E> type) {
        try {
            return Enum.valueOf(type, raw);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("property '" + key + "': unknown value '" + raw + "'; allowed values: "
                    + Arrays.toString(type.getEnumConstants()), e);
        }
    }

    private static String trimmed(Properties p, String key) {
        String raw = p.getProperty(key);
        return raw == null ? null : raw.trim();
    }
}
