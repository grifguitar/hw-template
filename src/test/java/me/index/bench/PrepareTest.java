package me.index.bench;

import me.index.bench.data.BinFiles;
import me.index.bench.data.Queries;
import me.index.bench.data.QueryOrder;
import me.index.bench.index.NetModel;
import me.index.bench.index.NetSpec;
import me.index.bench.index.Trainer;
import me.index.config.parameters.enums.DataSize;
import me.index.config.parameters.enums.Keyset;
import me.index.config.parameters.enums.LossFunction;
import me.index.config.parameters.enums.Workload;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrepareTest {
    private static final PrintStream SILENT = new PrintStream(OutputStream.nullOutputStream());
    private static final Trainer.Settings TRAINING = new Trainer.Settings(2, 1000, 32, LossFunction._squared, Double.NaN, 42);

    private static Prepare.Settings settings(Path work, Keyset keyset, DataSize size, List<NetSpec> nets) {
        return new Prepare.Settings(work, keyset, size, Path.of(""), 42, 3000,
                List.of(Workload._uniform, Workload._zipf), List.of(QueryOrder.random, QueryOrder.sorted), nets, TRAINING);
    }

    @Test
    void writesKeysQueriesAndModelsThatFitTogether(@TempDir Path work) throws IOException {
        List<NetSpec> nets = List.of(NetSpec.parse("_relu:4x4"), NetSpec.parse("_relu:none"));
        Prepare.Settings s = settings(work, Keyset._gauss_int64, DataSize._1e4, nets);
        Path dir = Prepare.run(s, SILENT);
        assertEquals(Prepare.directory(work, Keyset._gauss_int64, DataSize._1e4), dir);
        assertEquals(work.resolve("_gauss_int64").resolve("_1e4"), dir);

        long[] keys = BinFiles.readKeys(dir.resolve(Prepare.KEYS_FILE));
        assertArrayEquals(Prepare.loadKeys(s), keys);

        for (Workload w : s.workloads()) {
            Queries random = BinFiles.readQueries(dir.resolve(Prepare.queriesFileName(w, QueryOrder.random)));
            Queries sorted = BinFiles.readQueries(dir.resolve(Prepare.queriesFileName(w, QueryOrder.sorted)));
            assertEquals(3000, random.size());
            random.requireConsistentWith(keys);
            sorted.requireConsistentWith(keys);
            int[] expected = random.expected().clone();
            Arrays.sort(expected);
            assertArrayEquals(expected, sorted.expected(), "both orders must hold the same lookups");
        }

        assertEquals(List.of("nn_relu_4x4", "nn_linear"), Files.readAllLines(dir.resolve(Prepare.MODELS_LIST)));
        Queries q = BinFiles.readQueries(dir.resolve(Prepare.queriesFileName(Workload._zipf, QueryOrder.random)));
        for (NetSpec spec : nets) {
            NetModel model = NetModel.read(dir.resolve(Prepare.modelFileName(spec)));
            assertEquals(spec, model.spec());
            Measure.verify(model.index(keys), q);
        }
        List<String> csv = Files.readAllLines(dir.resolve(Prepare.MODELS_CSV));
        assertEquals(3, csv.size());
        assertTrue(csv.get(0).startsWith("index,activation,layers"));
        assertTrue(csv.get(1).startsWith("nn_relu_4x4,_relu,4x4,33,_squared,2,1000,"), csv.get(1));
    }

    @Test
    void fileNamesFollowTheConventionBenchShUses() {
        assertEquals("queries_uniform_random.bin", Prepare.queriesFileName(Workload._uniform, QueryOrder.random));
        assertEquals("queries_x_y_99_sorted.bin", Prepare.queriesFileName(Workload._x_y_99, QueryOrder.sorted));
        assertEquals("nn_leaky_relu_4x4.model", Prepare.modelFileName(NetSpec.parse("_leaky_relu:4x4")));
        assertEquals("keys.bin", Prepare.KEYS_FILE);
        assertEquals("models.txt", Prepare.MODELS_LIST);
    }

    @Test
    void preparationIsDeterministic(@TempDir Path tmp) throws IOException {
        Path a = Prepare.run(settings(tmp.resolve("a"), Keyset._log_norm_int64, DataSize._1e4, List.of()), SILENT);
        Path b = Prepare.run(settings(tmp.resolve("b"), Keyset._log_norm_int64, DataSize._1e4, List.of()), SILENT);
        for (String file : List.of(Prepare.KEYS_FILE, Prepare.queriesFileName(Workload._zipf, QueryOrder.random),
                Prepare.queriesFileName(Workload._uniform, QueryOrder.sorted))) {
            assertArrayEquals(Files.readAllBytes(a.resolve(file)), Files.readAllBytes(b.resolve(file)), file);
        }
        assertEquals(List.of(), Files.readAllLines(a.resolve(Prepare.MODELS_LIST)));
    }

    @Test
    void workloadsGetDifferentQueries(@TempDir Path work) throws IOException {
        Path dir = Prepare.run(settings(work, Keyset._gauss_int32, DataSize._1e4, List.of()), SILENT);
        Queries uniform = BinFiles.readQueries(dir.resolve(Prepare.queriesFileName(Workload._uniform, QueryOrder.random)));
        Queries zipf = BinFiles.readQueries(dir.resolve(Prepare.queriesFileName(Workload._zipf, QueryOrder.random)));
        assertTrue(!Arrays.equals(uniform.expected(), zipf.expected()));
    }

    @Test
    void invalidSettingsAreRejected(@TempDir Path work) {
        assertThrows(IllegalArgumentException.class,
                () -> Prepare.run(settings(work, Keyset._gauss_int32, DataSize._max, List.of()), SILENT));
        assertThrows(IOException.class,
                () -> Prepare.run(settings(work, Keyset._fb_200M_uint64, DataSize._1e4, List.of()), SILENT));
        assertThrows(IllegalArgumentException.class, () -> new Prepare.Settings(work, Keyset._gauss_int64,
                DataSize._1e4, Path.of(""), 1, 0, List.of(Workload._uniform), List.of(QueryOrder.random), List.of(), TRAINING));
        assertThrows(IllegalArgumentException.class, () -> new Prepare.Settings(work, Keyset._gauss_int64,
                DataSize._1e4, Path.of(""), 1, 10, List.of(), List.of(QueryOrder.random), List.of(), TRAINING));
        assertThrows(IllegalArgumentException.class, () -> new Prepare.Settings(work, Keyset._gauss_int64,
                DataSize._1e4, Path.of(""), 1, 10, List.of(Workload._uniform), List.of(), List.of(), TRAINING));
    }
}
