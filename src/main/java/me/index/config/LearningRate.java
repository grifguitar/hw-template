package me.index.config;

import java.util.Optional;
import java.util.Properties;

public record LearningRate(double value) {
    public static final String KEY = "train.learning.rate";

    public LearningRate {
        if (!(value > 0) || !Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "property '" + KEY + "': expected a finite positive number, got " + value);
        }
    }

    public static Optional<LearningRate> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(new LearningRate(Double.parseDouble(value.trim())));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "property '" + KEY + "': expected a number, got '" + value.trim() + "'", e);
        }
    }

    public static LearningRate defaultFor(LossFunction loss) {
        return new LearningRate(loss.defaultLearningRate());
    }
}
