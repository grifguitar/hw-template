package me.index.config.parameters.enums;

import me.index.config.Config;

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

    public static Optional<MaxErr> parse(Properties p) {
        return Config.parseEnum(p, KEY, values());
    }
}
