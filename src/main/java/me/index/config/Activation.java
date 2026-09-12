package me.index.config;

import me.index.ml.models.LeakyReluNet;
import me.index.ml.Loss;
import me.index.ml.Net;
import me.index.ml.models.ReluNet;
import me.index.ml.models.SigmoidNet;
import me.index.ml.models.SoftsignNet;
import me.index.ml.models.TanhNet;

import java.util.Random;

public enum Activation {
    _relu {
        @Override
        public Net create(int[] sz, double lr, int batch, Loss loss, Random rnd) {
            return new ReluNet(sz, lr, batch, loss, rnd);
        }
    },
    _leaky_relu {
        @Override
        public Net create(int[] sz, double lr, int batch, Loss loss, Random rnd) {
            return new LeakyReluNet(sz, lr, batch, loss, rnd);
        }
    },
    _sigmoid {
        @Override
        public Net create(int[] sz, double lr, int batch, Loss loss, Random rnd) {
            return new SigmoidNet(sz, lr, batch, loss, rnd);
        }
    },
    _tanh {
        @Override
        public Net create(int[] sz, double lr, int batch, Loss loss, Random rnd) {
            return new TanhNet(sz, lr, batch, loss, rnd);
        }
    },
    _softsign {
        @Override
        public Net create(int[] sz, double lr, int batch, Loss loss, Random rnd) {
            return new SoftsignNet(sz, lr, batch, loss, rnd);
        }
    };

    public abstract Net create(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd);
}
