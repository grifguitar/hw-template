package me.index;

import me.index.config.Config;
import me.index.config.DataSize;
import me.index.config.Keyset;
import me.index.io.ReadUtils;
import me.index.math.Utils;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

public final class Context {
    public final Config config;
    public final long[] keys;

    public Context(Config cfg, Path dataDir) throws IOException {
        this.config = cfg;
        Random rnd = cfg.seed().newRandom();
        Keyset keyset = cfg.keyset();

        if (!keyset.isLong && !keyset.isSOSD() && cfg.dataSize() == DataSize._max) {
            throw new IllegalArgumentException("incorrect properties: data.size=" + DataSize._max
                    + " is not available for 32-bit synthetic keysets (keyset=" + keyset + ")");
        }

        this.keys = switch (keyset.source) {
            case SOSD -> ReadUtils.read(dataDir.resolve(keyset.fileName()), cfg.dataSize().size,
                    keyset.isLong, keyset.needShift, keyset.needPlusOne);
            case GAUSSIAN -> Utils.genKeysGaussian(cfg.dataSize().size, keyset.isLong, rnd);
            case LOGNORMAL -> Utils.genKeysLognormal(cfg.dataSize().size, keyset.isLong, rnd);
        };

        if (keys.length < 2) {
            throw new IllegalStateException("keyset " + keyset + " with data.size=" + cfg.dataSize()
                    + " yielded " + keys.length + " distinct key(s); at least 2 are required");
        }
        for (int i = 1; i < keys.length; i++) {
            if (keys[i - 1] >= keys[i]) {
                throw new IllegalStateException("keyset " + keyset + " is not strictly increasing: keys[" + (i - 1)
                        + "] = " + keys[i - 1] + " >= keys[" + i + "] = " + keys[i]
                        + "; the pipeline treats the array index as the key's rank, so the input must be sorted"
                        + " and free of duplicates");
            }
        }
    }

    public static Context read(Path configFile, Path dataDir) throws IOException {
        return new Context(Config.load(configFile), dataDir);
    }
}
