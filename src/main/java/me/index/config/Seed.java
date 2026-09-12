package me.index.config;

import java.util.Optional;
import java.util.Properties;
import java.util.Random;

public record Seed(long value) {
    public static final String KEY = "train.seed";
    public static final Seed DEFAULT = new Seed(42L);

    public static Optional<Seed> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(new Seed(Long.parseLong(value.trim())));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "property '" + KEY + "': expected an integer, got '" + value.trim() + "'", e);
        }
    }

    public Random newRandom() {
        return new Random(value);
    }
}
