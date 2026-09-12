package me.index.config.parameters.enums;

import me.index.config.Config;

import java.util.Arrays;
import java.util.Optional;
import java.util.Properties;

public enum DataSize {
    _1e4(1e4), _1e5(1e5), _1e6(1e6), _1e7(1e7), _max(2e8);

    public static final String KEY = "data.size";

    public static Optional<DataSize> parse(Properties p) {
        return Config.parseEnum(p, KEY, values());
    }

    public static IllegalArgumentException except() {
        return new IllegalArgumentException("missing required property '" + KEY
                + "'; allowed values: " + Arrays.toString(values()));
    }

    public final int size;

    DataSize(double size) {
        this.size = (int) size;
    }
}
