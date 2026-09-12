package me.index.config.parameters.enums;

import me.index.config.Config;

import java.util.Optional;
import java.util.Properties;

public enum Workload {
    _zipf, _uniform, _x_y_9999, _x_y_99, _x_y_90;

    public static final String KEY = "workload.distribution";
    public static final Workload DEFAULT = _uniform;

    public static Optional<Workload> parse(Properties p) {
        return Config.parseEnum(p, KEY, values());
    }
}
