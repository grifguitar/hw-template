package me.index.config;

import me.index.config.parameters.enums.*;
import me.index.config.parameters.records.*;
import me.index.ml.Loss;
import me.index.ml.loss.SquaredLoss;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.RecordComponent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;

class ConfigTest {
    private static Properties minimal() {
        Properties p = new Properties();
        p.setProperty(Keyset.KEY, "_gauss_int32");
        p.setProperty(DataSize.KEY, "_1e4");
        return p;
    }

    @Test
    void missingOptionalKeysFallBackToDefaultsAndNeverToNull() {
        Config c = Config.read(minimal());
        assertEquals(Keyset._gauss_int32, c.keyset());
        assertEquals(DataSize._1e4, c.dataSize());
        assertEquals(Workload.DEFAULT, c.workload());
        assertEquals(WorkloadPerm.DEFAULT, c.workloadPerm());
        assertEquals(WorkloadFactor.DEFAULT, c.workloadFactor());
        assertEquals(MaxErr.DEFAULT, c.maxErr());
        assertEquals(Activation.DEFAULT, c.activation());
        assertEquals(LossFunction.DEFAULT, c.loss());
        assertEquals(Seed.DEFAULT, c.seed());
        assertEquals(Epochs.DEFAULT, c.epochs());
        assertEquals(BatchSize.DEFAULT, c.batchSize());
        assertEquals(HiddenLayers.DEFAULT, c.hiddenLayers());
        assertEquals(PlotPath.DEFAULT, c.plotPath());
    }

    @Test
    void missingRequiredKeyIsReportedWithAllowedValues() {
        Properties p = minimal();
        p.remove(Keyset.KEY);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.read(p));
        assertTrue(e.getMessage().contains("keyset"), e.getMessage());
        assertTrue(e.getMessage().contains("_gauss_int32"), e.getMessage());
    }

    @Test
    void unknownEnumValueIsReportedWithAllowedValues() {
        Properties p = minimal();
        p.setProperty(Workload.KEY, "_zipff");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.read(p));
        assertTrue(e.getMessage().contains("_zipff"), e.getMessage());
        assertTrue(e.getMessage().contains("_zipf"), e.getMessage());
    }

    @Test
    void trainingValuesAreParsedAndValidated() {
        Properties p = minimal();
        p.setProperty(Seed.KEY, "7");
        p.setProperty(LearningRate.KEY, "0.01");
        p.setProperty(BatchSize.KEY, "64");
        p.setProperty(Epochs.KEY, "5");
        p.setProperty(HiddenLayers.KEY, "16, 8 ,4");
        p.setProperty(PlotPath.KEY, "out/x.pdf");

        Config c = Config.read(p);
        assertEquals(7L, c.seed().value());
        assertEquals(0.01, c.learningRate().value());
        assertEquals(64, c.batchSize().value());
        assertEquals(5, c.epochs().value());
        assertEquals(List.of(16, 8, 4), c.hiddenLayers().sizes());
        assertEquals("out/x.pdf", c.plotPath().value());
        assertArrayEquals(new int[]{1, 16, 8, 4, 1}, c.hiddenLayers().layerSizes());
    }

    @Test
    void emptyHiddenLayersMeanADirectLinearModel() {
        Properties p = minimal();
        p.setProperty(HiddenLayers.KEY, "");
        assertArrayEquals(new int[]{1, 1}, Config.read(p).hiddenLayers().layerSizes());
    }

    @Test
    void invalidNumbersAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty(Epochs.KEY, "0");
            Config.read(p);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty(LearningRate.KEY, "-1");
            Config.read(p);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty(HiddenLayers.KEY, "4,x");
            Config.read(p);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            Properties p = minimal();
            p.setProperty(Seed.KEY, "abc");
            Config.read(p);
        });
    }

    @Test
    void valuesAreTrimmed() {
        Properties p = minimal();
        p.setProperty(Keyset.KEY, "  _gauss_int64  ");
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
            p.setProperty(LossFunction.KEY, lf.name());
            Config c = Config.read(p);
            assertEquals(lf, c.loss());
            assertEquals(lf.defaultParameter(), c.lossParameter().value(), lf.name());
            assertEquals(lf.defaultLearningRate(), c.learningRate().value(), lf.name());
            assertTrue(lf.defaultLearningRate() > 0 && Double.isFinite(lf.defaultLearningRate()), lf.name());
            assertDoesNotThrow(() -> c.lossParameter().newLoss(c.loss()), lf.name());
            assertNotNull(c.plotPath(), lf.name());
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
        p.setProperty(LossFunction.KEY, "_huber");
        p.setProperty(LearningRate.KEY, "0.001");
        assertEquals(0.001, Config.read(p).learningRate().value());
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
        p.setProperty(LossFunction.KEY, "_power");
        p.setProperty(LossParameter.KEY, "0.5");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.read(p));
        assertTrue(e.getMessage().contains("p must be finite and at least 1"), e.getMessage());
    }

    @Test
    void anExplicitLossParameterOverridesTheDefault() {
        Properties p = minimal();
        p.setProperty(LossFunction.KEY, "_huber");
        p.setProperty(LossParameter.KEY, "0.005");
        assertEquals(0.005, Config.read(p).lossParameter().value());
    }

    @Test
    void anEmptyConfigStillYieldsEveryOptionalParameter() {
        Properties p = minimal();
        Config c = Config.read(p);
        for (RecordComponent component : Config.class.getRecordComponents()) {
            assertDoesNotThrow(() -> assertNotNull(component.getAccessor().invoke(c), component.getName()));
        }
    }

    @Test
    void aDirectlyBuiltConfigStillValidatesTheLossParameter() {
        Config c = Config.read(minimal());
        assertThrows(IllegalArgumentException.class, () -> new Config(
                c.keyset(), c.dataSize(), c.workload(), c.workloadPerm(), c.workloadFactor(), c.maxErr(),
                c.activation(), LossFunction._power, new LossParameter(0.5), c.learningRate(),
                c.batchSize(), c.epochs(), c.hiddenLayers(), c.seed(), c.plotPath()));
    }

    @Test
    void aMisspelledPropertyIsRejectedInsteadOfIgnored() {
        Properties p = minimal();
        p.setProperty("net.hiden.layers", "8,8");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Config.read(p));
        assertTrue(e.getMessage().contains("net.hiden.layers"), e.getMessage());
        assertTrue(e.getMessage().contains(HiddenLayers.KEY), e.getMessage());
    }

    @Test
    void everyKeyOfTheShippedConfigIsOwnedByAType() throws IOException {
        Properties p = new Properties();
        try (var in = Files.newInputStream(Path.of("config.properties"))) {
            p.load(in);
        }
        for (String key : p.stringPropertyNames()) {
            assertTrue(Config.KEYS.contains(key), "config.properties declares an unowned key: " + key);
        }
        assertEquals(15, Config.KEYS.size(), "every property must have exactly one type answering for it");
    }

    @Test
    void onlySosdKeysetsNameADataFile() {
        for (Keyset k : Keyset.values()) {
            assertEquals(k.source == Keyset.Source.SOSD, k.isSOSD(), k.name());
            if (k.isSOSD()) {
                assertEquals(k.name().substring(1), k.fileName());
            } else {
                assertThrows(IllegalStateException.class, k::fileName);
                assertFalse(k.needShift);
                assertFalse(k.needPlusOne);
            }
        }
    }
}
