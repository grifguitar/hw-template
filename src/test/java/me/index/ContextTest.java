package me.index;

import me.index.config.Config;
import me.index.config.DataSize;
import me.index.config.Keyset;
import me.index.config.MaxErr;
import me.index.config.Training;
import me.index.config.Workload;
import me.index.config.WorkloadFactor;
import me.index.config.WorkloadPerm;
import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ContextTest {
    private static Config config(Keyset keyset, DataSize size) {
        return new Config(keyset, Workload._uniform, WorkloadPerm._false, WorkloadFactor._read_only,
                size, MaxErr._0,
                new Training(42L, 0.05, 32, 1, List.of(4), "plot.pdf"));
    }

    @Test
    void generatesSortedDistinctSyntheticKeys() throws IOException {
        Context ctx = new Context(config(Keyset._gauss_int32, DataSize._1e4), Path.of(""));
        assertTrue(ctx.keys.length > 1);
        for (int i = 1; i < ctx.keys.length; i++) {
            assertTrue(ctx.keys[i - 1] < ctx.keys[i]);
        }
    }

    @Test
    void maxSizeIsRejectedForThirtyTwoBitSyntheticKeysets() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new Context(config(Keyset._gauss_int32, DataSize._max), Path.of("")));
        assertTrue(e.getMessage().contains("_max"), e.getMessage());
    }

    @Test
    void missingSosdFileFailsWithThePathInTheMessage() {
        IOException e = assertThrows(IOException.class,
                () -> new Context(config(Keyset._fb_200M_uint64, DataSize._1e4), Path.of("no-such-dir")));
        assertTrue(e.getMessage().contains("fb_200M_uint64"), e.getMessage());
    }

    @Test
    void seedFromConfigDrivesGeneration() throws IOException {
        Config a = config(Keyset._gauss_int64, DataSize._1e4);
        Config b = new Config(a.keyset(), a.workload(), a.workloadPerm(), a.workloadFactor(), a.dataSize(),
                a.maxErr(), new Training(7L, 0.05, 32, 1, List.of(4), "plot.pdf"));
        assertArrayEquals(new Context(a, Path.of("")).keys, new Context(a, Path.of("")).keys);
        assertFalse(java.util.Arrays.equals(new Context(a, Path.of("")).keys, new Context(b, Path.of("")).keys));
    }

    @Test
    void unsortedKeysetIsRejected(@TempDir Path dir) throws IOException {
        Path f = dir.resolve("fb_200M_uint64");
        long[] vals = {5, 9, 5};
        ByteBuffer buf = ByteBuffer.allocate(8 + vals.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(vals.length);
        for (long v : vals) buf.putLong(v);
        Files.write(f, buf.array());

        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> new Context(config(Keyset._fb_200M_uint64, DataSize._1e4), dir));
        assertTrue(e.getMessage().contains("not strictly increasing"), e.getMessage());
    }

    @Test
    void readLoadsTheConfigFileAndTheKeyset(@TempDir Path dir) throws IOException {
        Path cfg = dir.resolve("c.properties");
        Files.writeString(cfg, "keyset=_log_norm_int64\ndata.size=_1e4\n");
        Context ctx = Context.read(cfg, dir);
        assertEquals(Keyset._log_norm_int64, ctx.config.keyset());
        assertTrue(ctx.keys.length > 1);
        for (int i = 1; i < ctx.keys.length; i++) {
            assertTrue(ctx.keys[i - 1] < ctx.keys[i]);
        }
    }
}
