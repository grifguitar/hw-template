package me.index;

import me.index.config.Config;
import me.index.io.ReadUtils;
import me.index.math.Utils;

import java.io.FileInputStream;
import java.util.Properties;
import java.util.Random;

public final class Context {
    public final long[] keys;

    public Context(Config cfg, String path) {
        Random rnd = new Random(42);

        if (!cfg.keyset().isLong && !cfg.keyset().isSOSD && cfg.dataSize().name().equals("_max")) {
            throw new RuntimeException("incorrect properties: max size for integer keys is not available");
        }

        if (cfg.keyset().isSOSD) {
            keys = ReadUtils.read(path + (cfg.keyset().name()).substring(1),
                    cfg.dataSize().size, cfg.keyset().isLong, cfg.keyset().needShift, cfg.keyset().needPlusOne);
        } else if (cfg.keyset().isGaussian) {
            keys = Utils.genKeysGaussian(cfg.dataSize().size, cfg.keyset().isLong, rnd);
        } else if (cfg.keyset().isLognormal) {
            keys = Utils.genKeysLognormal(cfg.dataSize().size, cfg.keyset().isLong, rnd);
        } else {
            throw new RuntimeException("unexpected case");
        }
    }

    public static Context read(String name, String path) {
        try (FileInputStream fis = new FileInputStream(name)) {
            Properties props = new Properties();
            props.load(fis);
            return new Context(Config.read(props), path);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
