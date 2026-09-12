package me.index.config.parameters.records;

import me.index.config.Config;

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

    public static Optional<BatchSize> parse(Properties p) {
        return Config.parseRecord(p, KEY, "a positive int", text -> new BatchSize(Integer.parseInt(text)));
    }
}
