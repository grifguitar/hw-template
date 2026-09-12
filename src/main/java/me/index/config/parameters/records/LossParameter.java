package me.index.config.parameters.records;

import me.index.config.parameters.enums.LossFunction;
import me.index.ml.Loss;

import java.util.Optional;
import java.util.Properties;

public record LossParameter(double value) {
    public static final String KEY = "train.loss.parameter";

    public LossParameter {
        if (!(value > 0) || !Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "property '" + KEY + "': expected a finite positive number, got " + value);
        }
    }

    public static Optional<LossParameter> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(new LossParameter(Double.parseDouble(value.trim())));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "property '" + KEY + "': expected a number, got '" + value.trim() + "'", e);
        }
    }

    public static LossParameter defaultFor(LossFunction loss) {
        return new LossParameter(loss.defaultParameter());
    }

    public void checkUsableWith(LossFunction loss) {
        loss.create(value);
    }

    public Loss newLoss(LossFunction loss) {
        return loss.create(value);
    }
}
