package me.index.config;

import me.index.ml.Loss;
import me.index.ml.loss.AbsoluteLoss;
import me.index.ml.loss.HuberLoss;
import me.index.ml.loss.LogCoshLoss;
import me.index.ml.loss.PowerLoss;
import me.index.ml.loss.SquaredLoss;

import java.util.Arrays;
import java.util.Optional;
import java.util.Properties;

public enum LossFunction {
    _squared {
        @Override
        public Loss create(double parameter) {
            return new SquaredLoss();
        }

        @Override
        public double defaultParameter() {
            return 1.0;
        }

        @Override
        public double defaultLearningRate() {
            return 0.05;
        }

        @Override
        public boolean usesParameter() {
            return false;
        }
    },
    _absolute {
        @Override
        public Loss create(double parameter) {
            return new AbsoluteLoss();
        }

        @Override
        public double defaultParameter() {
            return 1.0;
        }

        @Override
        public double defaultLearningRate() {
            return 0.005;
        }

        @Override
        public boolean usesParameter() {
            return false;
        }
    },
    _huber {
        @Override
        public Loss create(double parameter) {
            return new HuberLoss(parameter);
        }

        @Override
        public double defaultParameter() {
            return TRANSITION_WIDTH;
        }

        @Override
        public double defaultLearningRate() {
            return 0.5;
        }
    },
    _log_cosh {
        @Override
        public Loss create(double parameter) {
            return new LogCoshLoss(parameter);
        }

        @Override
        public double defaultParameter() {
            return TRANSITION_WIDTH;
        }

        @Override
        public double defaultLearningRate() {
            return 1.0;
        }
    },
    _power {
        @Override
        public Loss create(double parameter) {
            return new PowerLoss(parameter);
        }

        @Override
        public double defaultParameter() {
            return 2.0;
        }

        @Override
        public double defaultLearningRate() {
            return _squared.defaultLearningRate();
        }
    };

    public static final String KEY = "train.loss";
    public static final LossFunction DEFAULT = _squared;

    public static final double TRANSITION_WIDTH = 0.02;

    public static Optional<LossFunction> read(Properties p) {
        String value = p.getProperty(KEY);
        if (value == null) {
            return Optional.empty();
        }
        String name = value.trim();
        for (LossFunction candidate : values()) {
            if (candidate.name().equals(name)) {
                return Optional.of(candidate);
            }
        }
        throw new IllegalArgumentException("property '" + KEY + "': unknown value '" + name
                + "'; allowed values: " + Arrays.toString(values()));
    }

    public abstract Loss create(double parameter);

    public abstract double defaultParameter();

    public abstract double defaultLearningRate();

    public boolean usesParameter() {
        return true;
    }
}
