package me.index.config;

import java.util.Arrays;
import java.util.Optional;
import java.util.Properties;

public enum WorkloadFactor {
    _read_only(1.0, 0.0), _real(0.8, 0.1), _write_heavy(0.33, 0.33);

    public static final String KEY = "workload.factor";
    public static final WorkloadFactor DEFAULT = _read_only;

    public final double read;
    public final double insert;

    WorkloadFactor(double read, double insert) {
        this.read = read;
        this.insert = insert;
    }

    public static Optional<WorkloadFactor> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        String name = value.trim();
        for (WorkloadFactor candidate : values()) {
            if (candidate.name().equals(name)) {
                return Optional.of(candidate);
            }
        }
        throw new IllegalArgumentException("property '" + KEY + "': unknown value '" + name
                + "'; allowed values: " + Arrays.toString(values()));
    }
}
