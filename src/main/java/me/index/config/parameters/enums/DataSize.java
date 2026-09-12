package me.index.config.parameters.enums;

import java.util.Arrays;
import java.util.Optional;
import java.util.Properties;

public enum DataSize {
    _1e4(1e4), _1e5(1e5), _1e6(1e6), _1e7(1e7), _max(2e8);

    public static final String KEY = "data.size";

    public static Optional<DataSize> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        String name = value.trim();
        for (DataSize candidate : values()) {
            if (candidate.name().equals(name)) {
                return Optional.of(candidate);
            }
        }
        throw new IllegalArgumentException("property '" + KEY + "': unknown value '" + name
                + "'; allowed values: " + Arrays.toString(values()));
    }

    public static IllegalArgumentException missing() {
        return new IllegalArgumentException("missing required property '" + KEY
                + "'; allowed values: " + Arrays.toString(values()));
    }

    public final int size;

    DataSize(double size) {
        this.size = (int) size;
    }
}
