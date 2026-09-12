package me.index.config.parameters.enums;

import java.util.Arrays;
import java.util.Optional;
import java.util.Properties;

public enum MaxErr {
    _0(0), _1(1), _2(2), _4(4), _8(8), _16(16), _32(32), _64(64);

    public static final String KEY = "max.err";
    public static final MaxErr DEFAULT = _0;

    public final int value;

    MaxErr(int value) {
        this.value = value;
    }

    public static Optional<MaxErr> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        String name = value.trim();
        for (MaxErr candidate : values()) {
            if (candidate.name().equals(name)) {
                return Optional.of(candidate);
            }
        }
        throw new IllegalArgumentException("property '" + KEY + "': unknown value '" + name
                + "'; allowed values: " + Arrays.toString(values()));
    }
}
