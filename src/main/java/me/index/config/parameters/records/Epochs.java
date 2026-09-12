package me.index.config.parameters.records;

import java.util.Optional;
import java.util.Properties;

public record Epochs(int value) {
    public static final String KEY = "train.epochs";
    public static final Epochs DEFAULT = new Epochs(100);

    public Epochs {
        if (value <= 0) {
            throw new IllegalArgumentException("property '" + KEY + "': expected a positive int, got " + value);
        }
    }

    public static Optional<Epochs> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(new Epochs(Integer.parseInt(value.trim())));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "property '" + KEY + "': expected a positive int, got '" + value.trim() + "'", e);
        }
    }
}
