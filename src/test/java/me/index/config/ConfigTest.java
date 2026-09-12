package me.index.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Properties;

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
