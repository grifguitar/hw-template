package me.index.config.parameters.enums;

import java.util.Arrays;
import java.util.Optional;
import java.util.Properties;

public enum Workload {
    _zipf, _uniform, _x_y_9999, _x_y_99, _x_y_90;

    public static final String KEY = "workload.distribution";
    public static final Workload DEFAULT = _uniform;

    public static Optional<Workload> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        String name = value.trim();
        for (Workload candidate : values()) {
            if (candidate.name().equals(name)) {
                return Optional.of(candidate);
            }
        }
        throw new IllegalArgumentException("property '" + KEY + "': unknown value '" + name
                + "'; allowed values: " + Arrays.toString(values()));
    }
}
