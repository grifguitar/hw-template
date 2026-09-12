package me.index.config.parameters.records;

import me.index.config.Config;
import me.index.config.parameters.enums.LossFunction;

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

    public static Optional<LearningRate> parse(Properties p) {
        return Config.parseRecord(p, KEY, "a number", text -> new LearningRate(Double.parseDouble(text)));
    }

    public static LearningRate defaultFor(LossFunction loss) {
        return new LearningRate(loss.defaultLearningRate());
    }
}
