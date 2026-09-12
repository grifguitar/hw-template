package me.index.config.parameters.records;

import me.index.config.Config;

import java.util.Optional;
import java.util.Properties;
import java.util.Random;

public record Seed(long value) {
    public static final String KEY = "train.seed";
    public static final Seed DEFAULT = new Seed(42L);

    public static Optional<Seed> parse(Properties p) {
        return Config.parseRecord(p, KEY, "an integer", text -> new Seed(Long.parseLong(text)));
    }

    public Random newRandom() {
        return new Random(value);
    }
}
