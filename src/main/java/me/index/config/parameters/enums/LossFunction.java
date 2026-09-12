package me.index.config.parameters.enums;

import me.index.config.Config;

import me.index.ml.Loss;
import me.index.ml.loss.AbsoluteLoss;
import me.index.ml.loss.HuberLoss;
import me.index.ml.loss.LogCoshLoss;
import me.index.ml.loss.PowerLoss;
import me.index.ml.loss.SquaredLoss;

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
    },
    _huber {
        @Override
        public Loss create(double parameter) {
            return new HuberLoss(parameter);
        }

        @Override
        public double defaultParameter() {
            return 0.02;
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
            return 0.02;
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
            return 0.05;
        }
    };

    public static final String KEY = "train.loss";
    public static final LossFunction DEFAULT = _squared;

    public static Optional<LossFunction> parse(Properties p) {
        return Config.parseEnum(p, KEY, values());
    }

    public abstract Loss create(double parameter);

    public abstract double defaultParameter();

    public abstract double defaultLearningRate();
}
