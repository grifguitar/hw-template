package me.index.bench.index;

import me.index.config.parameters.enums.LossFunction;
import me.index.math.Utils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NetModelTest {
    private static final Trainer.Settings SETTINGS = new Trainer.Settings(2, 1000, 32, LossFunction._huber, 0.1, 9);

    private static long[] keys() {
        return Utils.genKeysGaussian(5000, true, new Random(1));
    }

    @Test
    void aWrittenModelReadsBackIdentically(@TempDir Path dir) throws IOException {
        long[] keys = keys();
        NetModel model = Trainer.train(keys, NetSpec.parse("_softsign:4x4"), SETTINGS);
        Path file = dir.resolve("m.model");
        model.write(file);
        NetModel read = NetModel.read(file);

        assertEquals(model.spec(), read.spec());
        assertArrayEquals(model.parameters(), read.parameters());
        assertEquals(model.n(), read.n());
        assertEquals(model.minKey(), read.minKey());
        assertEquals(model.maxKey(), read.maxKey());
        assertEquals(model.keysChecksum(), read.keysChecksum());
        assertEquals(model.errLo(), read.errLo());
        assertEquals(model.errHi(), read.errHi());
        assertEquals(model.meanAbsErr(), read.meanAbsErr());
        assertEquals("_huber", read.loss());
        assertEquals(2, read.epochs());
        assertEquals(1000, read.trainSample());
        assertEquals(model.trainMse(), read.trainMse());
        assertEquals(model.trainSeconds(), read.trainSeconds());

        NetIndex a = model.index(keys);
        NetIndex b = read.index(keys);
        for (long key : keys) {
            assertEquals(a.position(key), b.position(key));
        }
    }

    @Test
    void aModelRefusesAnotherKeyset() {
        long[] keys = keys();
        NetModel model = Trainer.train(keys, NetSpec.parse("_relu:4"), SETTINGS);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> model.index(Arrays.copyOf(keys, keys.length - 1)));
        assertTrue(e.getMessage().contains("different keyset"), e.getMessage());

        long[] biggerMax = keys.clone();
        biggerMax[keys.length - 1]++;
        assertThrows(IllegalArgumentException.class, () -> model.index(biggerMax));

        int gap = keys.length / 2;
        while (keys[gap] + 1 >= keys[gap + 1]) {
            gap++;
        }
        long[] sameEnds = keys.clone();
        sameEnds[gap]++;
        assertThrows(IllegalArgumentException.class, () -> model.index(sameEnds),
                "a keyset with the same size and ends but other keys must be detected by the checksum");
    }

    @Test
    void theTrainingSampleIsCappedByTheKeyset() {
        long[] keys = {1, 5, 9, 100, 1000};
        NetModel model = Trainer.train(keys, NetSpec.parse("_relu:none"), SETTINGS);
        assertEquals(5, model.trainSample());
        assertEquals(5, model.n());
    }

    @Test
    void corruptModelFilesAreReported(@TempDir Path dir) throws IOException {
        Path garbage = dir.resolve("garbage");
        Files.write(garbage, new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        assertTrue(assertThrows(IOException.class, () -> NetModel.read(garbage)).getMessage()
                .contains("not a model file"));

        NetModel model = Trainer.train(keys(), NetSpec.parse("_relu:4"), SETTINGS);
        Path good = dir.resolve("good");
        model.write(good);
        byte[] bytes = Files.readAllBytes(good);

        Path truncated = dir.resolve("truncated");
        Files.write(truncated, Arrays.copyOf(bytes, bytes.length - 10));
        assertTrue(assertThrows(IOException.class, () -> NetModel.read(truncated)).getMessage().contains("truncated"));

        Path trailing = dir.resolve("trailing");
        Files.write(trailing, Arrays.copyOf(bytes, bytes.length + 1));
        assertTrue(assertThrows(IOException.class, () -> NetModel.read(trailing)).getMessage().contains("trailing"));

        assertThrows(IOException.class, () -> NetModel.read(dir.resolve("absent")));
    }

    @Test
    void theConstructorValidatesTheModel() {
        NetSpec spec = NetSpec.parse("_relu:none");
        assertThrows(IllegalArgumentException.class,
                () -> new NetModel(spec, new double[3], 10, 0, 9, 0, 0, 0, 0, "_squared", 1, 1, 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new NetModel(spec, new double[2], 1, 0, 9, 0, 0, 0, 0, "_squared", 1, 1, 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new NetModel(spec, new double[2], 10, 9, 9, 0, 0, 0, 0, "_squared", 1, 1, 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new NetModel(spec, new double[2], 10, 0, 9, 0, 10, 0, 0, "_squared", 1, 1, 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new NetModel(spec, new double[2], 10, 0, 9, 0, 0, -1, 0, "_squared", 1, 1, 0, 0));
    }

    @Test
    void parametersAreDefensivelyCopied() {
        double[] params = {1, 2};
        NetModel model = new NetModel(NetSpec.parse("_relu:none"), params, 10, 0, 9, 0, 0, 0, 0, "_squared", 1, 1, 0, 0);
        params[0] = 42;
        assertEquals(1.0, model.parameters()[0]);
        model.parameters()[1] = 42;
        assertEquals(2.0, model.parameters()[1]);
        assertEquals(1, model.window());
    }

    @Test
    void trainerSettingsAreValidated() {
        assertThrows(IllegalArgumentException.class, () -> new Trainer.Settings(0, 10, 1, LossFunction._squared, Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> new Trainer.Settings(1, 1, 1, LossFunction._squared, Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> new Trainer.Settings(1, 10, 0, LossFunction._squared, Double.NaN, 1));
        assertThrows(IllegalArgumentException.class, () -> new Trainer.Settings(1, 10, 1, LossFunction._squared, -1, 1));
        assertThrows(NullPointerException.class, () -> new Trainer.Settings(1, 10, 1, null, Double.NaN, 1));
        assertEquals(LossFunction._huber.defaultLearningRate(),
                new Trainer.Settings(1, 10, 1, LossFunction._huber, Double.NaN, 1).effectiveLearningRate());
        assertEquals(0.3, new Trainer.Settings(1, 10, 1, LossFunction._huber, 0.3, 1).effectiveLearningRate());
    }
}
