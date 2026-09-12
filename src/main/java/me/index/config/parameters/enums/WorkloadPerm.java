package me.index.config.parameters.enums;

import java.util.Arrays;
import java.util.Optional;
import java.util.Properties;

public enum WorkloadPerm {
    _true(true), _false(false);

    public static final String KEY = "workload.permutation";
    public static final WorkloadPerm DEFAULT = _false;

    public final boolean value;

    WorkloadPerm(boolean value) {
        this.value = value;
    }

    public static Optional<WorkloadPerm> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        String name = value.trim();
        for (WorkloadPerm candidate : values()) {
            if (candidate.name().equals(name)) {
                return Optional.of(candidate);
            }
        }
        throw new IllegalArgumentException("property '" + KEY + "': unknown value '" + name
                + "'; allowed values: " + Arrays.toString(values()));
    }
}
