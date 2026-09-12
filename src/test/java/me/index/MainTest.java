package me.index;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class MainTest {

    private record Run(String out, String error) {
        boolean failed() {
            return error != null;
        }
    }

    private static Run execute(Path configFile) {
        PrintStream oldOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8));
            Main.run(configFile, Path.of(""));
            return new Run(captured.toString(StandardCharsets.UTF_8), null);
        } catch (IOException | IllegalArgumentException | IllegalStateException e) {
            return new Run(captured.toString(StandardCharsets.UTF_8), String.valueOf(e.getMessage()));
        } finally {
            System.setOut(oldOut);
        }
    }

    private static Path config(Path dir, String name, String body) throws IOException {
        Path f = dir.resolve(name);
        Files.writeString(f, body);
        return f;
    }

    private static String escape(Path p) {
        return p.toString().replace("\\", "\\\\");
    }

    @Test
    void endToEndRunReportsMetricsAndWritesThePlot(@TempDir Path dir) throws IOException {
        Path plot = dir.resolve("out.pdf");
        Path cfg = config(dir, "ok.properties", """
                keyset=_gauss_int32
                data.size=_1e4
                train.epochs=3
                net.hidden.layers=4,4
                plot.path=%s
                """.formatted(escape(plot)));

        Run r = execute(cfg);
        assertFalse(r.failed(), r.error());
        assertTrue(r.out().contains("keyset=_gauss_int32"), r.out());
        assertTrue(r.out().contains("distinct keys:"), r.out());
        assertTrue(r.out().contains("maxErr:"), r.out());
        assertTrue(Files.exists(plot), "plot file must be written");
        assertTrue(Files.readString(plot, StandardCharsets.ISO_8859_1).startsWith("%PDF-1.4"));
    }

    @Test
    void runIsReproducibleForAFixedSeed(@TempDir Path dir) throws IOException {
        String body = """
                keyset=_gauss_int32
                data.size=_1e4
                train.epochs=3
                plot.path=%s
                """;
        Path p1 = dir.resolve("a.pdf");
        Path p2 = dir.resolve("b.pdf");
        Run first = execute(config(dir, "a.properties", body.formatted(escape(p1))));
        Run second = execute(config(dir, "b.properties", body.formatted(escape(p2))));

        assertFalse(first.failed(), first.error());
        assertEquals(maxErrOf(first.out()), maxErrOf(second.out()));
        assertArrayEquals(Files.readAllBytes(p1), Files.readAllBytes(p2), "same seed must give the same PDF");
    }

    @Test
    void aDifferentSeedChangesTheResult(@TempDir Path dir) throws IOException {
        String body = """
                keyset=_gauss_int32
                data.size=_1e4
                train.epochs=3
                train.seed=%d
                plot.path=%s
                """;
        Run a = execute(config(dir, "s1.properties", body.formatted(1, escape(dir.resolve("s1.pdf")))));
        Run b = execute(config(dir, "s2.properties", body.formatted(2, escape(dir.resolve("s2.pdf")))));
        assertFalse(a.failed(), a.error());
        assertFalse(b.failed(), b.error());
        assertNotEquals(maxErrOf(a.out()), maxErrOf(b.out()));
    }

    @Test
    void divergedTrainingIsReportedInsteadOfWritingABrokenPlot(@TempDir Path dir) throws IOException {
        Path plot = dir.resolve("diverged.pdf");
        Path cfg = config(dir, "diverge.properties", """
                keyset=_gauss_int32
                data.size=_1e4
                train.learning.rate=1e9
                train.epochs=5
                net.hidden.layers=8,8
                plot.path=%s
                """.formatted(escape(plot)));

        Run r = execute(cfg);
        assertTrue(r.failed(), "diverged training must fail the run");
        assertTrue(r.error().contains("diverged"), r.error());
        assertFalse(Files.exists(plot), "no PDF must be produced for a diverged run");
    }

    @Test
    void missingConfigIsReported(@TempDir Path dir) {
        Run r = execute(dir.resolve("absent.properties"));
        assertTrue(r.failed());
        assertTrue(r.error().contains("absent.properties"), r.error());
    }

    @Test
    void previewElidesLongKeysetsAndPrintsShortOnesInFull() {
        assertEquals("1 2", Main.preview(new long[]{1, 2}));
        assertEquals("1 2 3 4 5 6", Main.preview(new long[]{1, 2, 3, 4, 5, 6}));
        assertEquals("1 2 3 ... 5 6 7", Main.preview(new long[]{1, 2, 3, 4, 5, 6, 7}));
    }

    private static long maxErrOf(String stdout) {
        return stdout.lines()
                .filter(l -> l.startsWith("maxErr:"))
                .map(l -> Long.parseLong(l.substring("maxErr:".length()).trim()))
                .findFirst().orElseThrow();
    }
}
