package me.index.bench.report;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ResultsCsvTest {
    @Test
    void appendedRowsReadBackIdenticallyUnderASingleHeader(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("nested").resolve("results.csv");
        ResultRow a = ReportTest.row("_gauss_int64", "_1e5", "_uniform", "random", "binary", 1, 123.25);
        ResultRow b = ReportTest.row("_fb_200M_uint64", "_max", "_x_y_99", "sorted", "nn_relu_4x4", 3, 1e-3);
        ResultsCsv.append(file, a);
        ResultsCsv.append(file, b);

        List<String> lines = Files.readAllLines(file);
        assertEquals(3, lines.size());
        assertEquals(String.join(",", ResultsCsv.COLUMNS), lines.getFirst());
        assertEquals(List.of(a, b), ResultsCsv.read(file));
        assertTrue(lines.get(1).contains("NaN"), "missing model time must be written as NaN");
    }

    @Test
    void columnsAreFoundByNameAndBlankLinesAreSkipped(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("r.csv");
        ResultRow a = ReportTest.row("_gauss_int64", "_1e5", "_uniform", "random", "binary", 1, 5);
        ResultsCsv.append(file, a);
        Files.writeString(file, Files.readString(file) + "\n\n");
        assertEquals(List.of(a), ResultsCsv.read(file));
    }

    @Test
    void textFieldsMustNotBreakTheCsv() {
        ResultRow ok = ReportTest.row("_gauss_int64", "_1e5", "_uniform", "random", "binary", 1, 5);
        assertThrows(IllegalArgumentException.class, () -> new ResultRow("a,b", ok.size(), ok.n(), ok.workload(),
                ok.order(), ok.queries(), ok.index(), ok.kind(), ok.activation(), ok.layers(), ok.params(), ok.fork(),
                ok.runs(), ok.warmupPasses(), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, "1"));
        assertThrows(IllegalArgumentException.class, () -> new ResultRow(ok.dataset(), ok.size(), ok.n(), ok.workload(),
                ok.order(), ok.queries(), "", ok.kind(), ok.activation(), ok.layers(), ok.params(), ok.fork(),
                ok.runs(), ok.warmupPasses(), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, "1"));
        assertEquals("1.500;2.000", ResultRow.joinRuns(new double[]{1.5, 2}));
    }

    @Test
    void malformedFilesAreReportedWithTheirLine(@TempDir Path dir) throws IOException {
        Path empty = dir.resolve("empty.csv");
        Files.writeString(empty, "");
        assertThrows(IOException.class, () -> ResultsCsv.read(empty));

        Path noColumn = dir.resolve("nocol.csv");
        Files.writeString(noColumn, "dataset,size\n_a,_b\n");
        assertTrue(assertThrows(IOException.class, () -> ResultsCsv.read(noColumn)).getMessage().contains("column"));

        Path good = dir.resolve("good.csv");
        ResultsCsv.append(good, ReportTest.row("_gauss_int64", "_1e5", "_uniform", "random", "binary", 1, 5));
        String header = Files.readAllLines(good).getFirst();
        String line = Files.readAllLines(good).get(1);

        Path shortLine = dir.resolve("short.csv");
        Files.writeString(shortLine, header + "\n" + line.substring(0, line.lastIndexOf(',')) + "\n");
        assertTrue(assertThrows(IOException.class, () -> ResultsCsv.read(shortLine)).getMessage().contains(":2:"));

        Path badNumber = dir.resolve("bad.csv");
        Files.writeString(badNumber, header + "\n" + line.replace(",100000,", ",many,") + "\n");
        assertTrue(assertThrows(IOException.class, () -> ResultsCsv.read(badNumber)).getMessage().contains(":2:"));
    }
}
