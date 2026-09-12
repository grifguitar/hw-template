package me.index.config.parameters.records;

import java.util.Optional;
import java.util.Properties;

public record BatchSize(int value) {
    public static final String KEY = "train.batch.size";
    public static final BatchSize DEFAULT = new BatchSize(32);

    public BatchSize {
        if (value <= 0) {
            throw new IllegalArgumentException("property '" + KEY + "': expected a positive int, got " + value);
        }
    }

    public static Optional<BatchSize> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(new BatchSize(Integer.parseInt(value.trim())));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "property '" + KEY + "': expected a positive int, got '" + value.trim() + "'", e);
        }
    }
}
