package me.index.config;

import me.index.ml.Loss;
import me.index.ml.loss.SquaredLoss;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class ConfigTest {
    private static Properties minimal() {
        Properties p = new Properties();
        p.setProperty("keyset", "_gauss_int32");
        p.setProperty("data.size", "_1e4");
        return p;
    }

    @Test
    void missingOptionalKeysFallBackToDefaultsAndNeverToNull() {
        Config c = Config.read(minimal());
        assertEquals(Keyset._gauss_int32, c.keyset());
        assertEquals(DataSize._1e4, c.dataSize());
        assertEquals(Config.DEFAULT_WORKLOAD, c.workload());
        assertEquals(Config.DEFAULT_WORKLOAD_PERM, c.workloadPerm());
        assertEquals(Config.DEFAULT_WORKLOAD_FACTOR, c.workloadFactor());
        assertEquals(Config.DEFAULT_MAX_ERR, c.maxErr());
        assertNotNull(c.training());
        assertEquals(Training.DEFAULT_SEED, c.training().seed());
        assertEquals(Training.DEFAULT_EPOCHS, c.training().epochs());
        assertEquals(Training.DEFAULT_HIDDEN_LAYERS, c.training().hiddenLayers());
        assertEquals(Training.DEFAULT_PLOT_PATH, c.training().plotPath());
    }

    @Test
    void missingRequiredKeyIsReportedWithAllowedValues() {
        Properties p = minimal();
        p.remove("keyset");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.read(p));
        assertTrue(e.getMessage().contains("keyset"), e.getMessage());
        assertTrue(e.getMessage().contains("_gauss_int32"), e.getMessage());
    }

    @Test
    void unknownEnumValueIsReportedWithAllowedValues() {
        Properties p = minimal();
        p.setProperty("workload.distribution", "_zipff");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.read(p));
        assertTrue(e.getMessage().contains("_zipff"), e.getMessage());
        assertTrue(e.getMessage().contains("_zipf"), e.getMessage());
    }

    @Test
    void trainingValuesAreParsedAndValidated() {
        Properties p = minimal();
        p.setProperty("train.seed", "7");
        p.setProperty("train.learning.rate", "0.01");
        p.setProperty("train.batch.size", "64");
        p.setProperty("train.epochs", "5");
        p.setProperty("net.hidden.layers", "16, 8 ,4");
        p.setProperty("plot.path", "out/x.pdf");

        Training t = Config.read(p).training();
        assertEquals(7L, t.seed());
        assertEquals(0.01, t.learningRate());
        assertEquals(64, t.batchSize());
        assertEquals(5, t.epochs());
        assertEquals(List.of(16, 8, 4), t.hiddenLayers());
        assertEquals("out/x.pdf", t.plotPath());
        assertArrayEquals(new int[]{1, 16, 8, 4, 1}, t.layerSizes());
    }

    @Test
    void emptyHiddenLayersMeanADirectLinearModel() {
        Properties p = minimal();
        p.setProperty("net.hidden.layers", "");
        assertArrayEquals(new int[]{1, 1}, Config.read(p).training().layerSizes());
    }

    @Test
    void invalidNumbersAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty("train.epochs", "0");
            Config.read(p);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty("train.learning.rate", "-1");
            Config.read(p);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty("net.hidden.layers", "4,x");
            Config.read(p);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty("train.seed", "abc");
            Config.read(p);
        });
    }

    @Test
    void valuesAreTrimmed() {
        Properties p = minimal();
        p.setProperty("keyset", "  _gauss_int64  ");
        assertEquals(Keyset._gauss_int64, Config.read(p).keyset());
    }

    @Test
    void loadReadsAFileAndNamesItInErrors(@TempDir Path dir) throws IOException {
        Path good = dir.resolve("good.properties");
        Files.writeString(good, "keyset=_gauss_int32\ndata.size=_1e4\n");
        assertEquals(Keyset._gauss_int32, Config.load(good).keyset());

        Path bad = dir.resolve("bad.properties");
        Files.writeString(bad, "keyset=_nope\ndata.size=_1e4\n");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.load(bad));
        assertTrue(e.getMessage().contains("bad.properties"), e.getMessage());

        assertThrows(IOException.class, () -> Config.load(dir.resolve("absent.properties")));
    }

    @Test
    void repositoryConfigIsValid() throws IOException {
        Path repoConfig = Path.of("config.properties");
        assertTrue(Files.isReadable(repoConfig), "config.properties must live at the repo root");
        assertNotNull(Config.load(repoConfig));
    }

    @Test
    void everyLossGetsItsOwnDefaultParameterAndLearningRate() {
        for (LossFunction lf : LossFunction.values()) {
            Properties p = minimal();
            p.setProperty("train.loss", lf.name());
            Config c = Config.read(p);
            assertEquals(lf, c.training().lossFunction());
            assertEquals(lf.defaultParameter(), c.training().lossParameter(), lf.name());
            assertEquals(lf.defaultLearningRate(), c.training().learningRate(), lf.name());
            assertTrue(lf.defaultLearningRate() > 0 && Double.isFinite(lf.defaultLearningRate()), lf.name());
            assertDoesNotThrow(() -> c.training().newNet(new Random(1)), lf.name());
        }
    }

    @Test
    void lossesThatNeedDifferentRatesActuallyGetThem() {
        assertTrue(LossFunction._absolute.defaultLearningRate() < LossFunction._squared.defaultLearningRate(),
                "the absolute loss must step more cautiously than the squared one");
        assertTrue(LossFunction._huber.defaultLearningRate() > LossFunction._squared.defaultLearningRate(),
                "a clipped loss must be allowed to step further than the squared one");
        assertTrue(LossFunction._log_cosh.defaultLearningRate() > LossFunction._squared.defaultLearningRate(),
                "the other clipped loss must too");
        assertEquals(LossFunction._squared.defaultLearningRate(), LossFunction._power.defaultLearningRate());
    }

    @Test
    void anExplicitLearningRateOverridesThePerLossDefault() {
        Properties p = minimal();
        p.setProperty("train.loss", "_huber");
        p.setProperty("train.learning.rate", "0.001");
        assertEquals(0.001, Config.read(p).training().learningRate());
    }

    @Test
    void robustLossDefaultsDoNotCollapseIntoTheSquaredLoss() {
        for (LossFunction lf : List.of(LossFunction._huber, LossFunction._log_cosh)) {
            assertTrue(lf.defaultParameter() < 0.5,
                    lf + " default parameter " + lf.defaultParameter() + " never reaches the linear branch");
            Loss robust = lf.create(lf.defaultParameter());
            Loss squared = new SquaredLoss();
            assertNotEquals(squared.gradient(0.5, 0.0), robust.gradient(0.5, 0.0), 1e-6, lf.name());
        }
        assertTrue(LossFunction._squared.defaultParameter() > 0, "defaults must stay usable as a positive double");
        assertFalse(LossFunction._squared.usesParameter());
        assertFalse(LossFunction._absolute.usesParameter());
        assertTrue(LossFunction._huber.usesParameter());
    }

    @Test
    void anUnusableLossParameterIsRejectedWhileReadingTheConfig() {
        Properties p = minimal();
        p.setProperty("train.loss", "_power");
        p.setProperty("train.loss.parameter", "0.5");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.read(p));
        assertTrue(e.getMessage().contains("p must be finite and at least 1"), e.getMessage());
    }

    @Test
    void anExplicitLossParameterOverridesTheDefault() {
        Properties p = minimal();
        p.setProperty("train.loss", "_huber");
        p.setProperty("train.loss.parameter", "0.005");
        assertEquals(0.005, Config.read(p).training().lossParameter());
    }

    @Test
    void keysetFlagsAreConsistentWithSource() {
        for (Keyset k : Keyset.values()) {
            assertEquals(k.source == Keyset.Source.SOSD, k.isSOSD, k.name());
            assertEquals(k.source == Keyset.Source.GAUSSIAN, k.isGaussian, k.name());
            assertEquals(k.source == Keyset.Source.LOGNORMAL, k.isLognormal, k.name());
            if (k.isSOSD) {
                assertEquals(k.name().substring(1), k.fileName());
            } else {
                assertThrows(IllegalStateException.class, k::fileName);
                assertFalse(k.needShift);
                assertFalse(k.needPlusOne);
            }
        }
    }
}
