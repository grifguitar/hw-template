package me.index.config.parameters.enums;

import me.index.config.Config;

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

    public static Optional<WorkloadPerm> parse(Properties p) {
        return Config.parseEnum(p, KEY, values());
    }
}
