package me.index.config.parameters.enums;

import me.index.config.Config;

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

    public static Optional<WorkloadFactor> parse(Properties p) {
        return Config.parseEnum(p, KEY, values());
    }
}
