package me.index.bench.report;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReportTest {
    static ResultRow row(String dataset, String size, String workload, String order, String index, int fork,
                         double ns) {
        boolean net = index.startsWith("nn_");
        return new ResultRow(dataset, size, size.equals("_1e5") ? 100_000 : 1_000_000, workload, order, 1000,
                index, net ? ResultRow.NET : ResultRow.BASELINE, net ? "_relu" : "-", net ? "4x4" : "-", net ? 33 : 0,
                fork, 10, 5, ns, ns * 0.9, ns * 1.1, ns, 1.0, ns, ns * 1.2, ns * 2, ns * 3,
                net ? ns / 2 : Double.NaN, net ? 3 : 0, net ? 4 : 0, net ? 8 : 100_000, net ? 1.5 : Double.NaN,
                net ? 328 : 0, 0, 0, 12345, "1.000;2.000");
    }

    private static List<ResultRow> rows() {
        List<ResultRow> rows = new ArrayList<>();
        double[] binary1e5 = {100, 120, 90};
        double[] net1e5 = {50, 40, 60};
        for (int fork = 1; fork <= 3; fork++) {
            rows.add(row("_gauss_int64", "_1e5", "_uniform", "random", "binary", fork, binary1e5[fork - 1]));
            rows.add(row("_gauss_int64", "_1e5", "_uniform", "random", "nn_relu_4x4", fork, net1e5[fork - 1]));
            rows.add(row("_gauss_int64", "_1e6", "_uniform", "random", "binary", fork, 300));
            rows.add(row("_gauss_int64", "_1e6", "_uniform", "random", "nn_relu_4x4", fork, 100));
            rows.add(row("_gauss_int64", "_1e6", "_uniform", "random", "branchless", fork, 250));
        }
        return rows;
    }

    private static Report.Summary find(List<Report.Summary> list, String size, String index) {
        return list.stream().filter(s -> s.size().equals(size) && s.index().equals(index)).findFirst().orElseThrow();
    }

    @Test
    void summariesTakeTheMedianOverForksAndTheSpeedupOverBinary() {
        List<Report.Summary> s = Report.summarize(rows());
        assertEquals(5, s.size());

        Report.Summary binary = find(s, "_1e5", "binary");
        assertEquals(100, binary.nsMedian());
        assertEquals(81, binary.nsBest(), 1e-9);
        assertEquals(0.3, binary.forkSpread(), 1e-9);
        assertEquals(1.0, binary.speedup());
        assertEquals(3, binary.forks());
        assertEquals(1e5, binary.sizeValue());

        Report.Summary net = find(s, "_1e5", "nn_relu_4x4");
        assertEquals(50, net.nsMedian());
        assertEquals(2.0, net.speedup(), 1e-12);
        assertEquals(25, net.modelNs());
        assertEquals(8, net.window());
        assertEquals(0.4, net.forkSpread(), 1e-9);

        assertEquals(3.0, find(s, "_1e6", "nn_relu_4x4").speedup(), 1e-12);
        assertEquals(1.2, find(s, "_1e6", "branchless").speedup(), 1e-12);
        assertTrue(Double.isNaN(find(s, "_1e6", "binary").modelNs()));

        assertEquals(List.of("binary", "nn_relu_4x4", "binary", "branchless", "nn_relu_4x4"),
                s.stream().map(Report.Summary::index).toList(), "sorted by size, baselines first");
    }

    @Test
    void withoutBinaryTheSpeedupIsUnknown() {
        List<Report.Summary> s = Report.summarize(List.of(
                row("_gauss_int64", "_1e5", "_zipf", "sorted", "branchless", 1, 10)));
        assertTrue(Double.isNaN(s.getFirst().speedup()));
    }

    @Test
    void scenariosAreKeptApart() {
        List<ResultRow> rows = new ArrayList<>(rows());
        rows.add(row("_gauss_int64", "_1e5", "_uniform", "sorted", "binary", 1, 10));
        rows.add(row("_gauss_int64", "_1e5", "_uniform", "sorted", "nn_relu_4x4", 1, 20));
        rows.add(row("_log_norm_int64", "_1e5", "_uniform", "random", "binary", 1, 1000));
        List<Report.Summary> s = Report.summarize(rows);
        Report.Summary sortedNet = s.stream()
                .filter(x -> x.order().equals("sorted") && x.index().equals("nn_relu_4x4")).findFirst().orElseThrow();
        assertEquals(0.5, sortedNet.speedup(), 1e-12);
        assertEquals(8, s.size());
    }

    @Test
    void writeProducesTablesChartsAndWarnings(@TempDir Path dir) throws IOException {
        List<ResultRow> rows = new ArrayList<>(rows());
        ResultRow allocating = row("_gauss_int64", "_1e6", "_uniform", "random", "binary", 4, 300);
        rows.add(new ResultRow(allocating.dataset(), allocating.size(), allocating.n(), allocating.workload(),
                allocating.order(), allocating.queries(), allocating.index(), allocating.kind(), allocating.activation(),
                allocating.layers(), allocating.params(), allocating.fork(), allocating.runs(), allocating.warmupPasses(),
                300, 290, 310, 300, 1, 300, 310, 320, 330, Double.NaN, 0, 0, 1_000_000, Double.NaN, 0,
                4096, 17, 1, "300.000"));

        List<Path> charts = Report.write(rows, dir, "results.csv");
        List<String> names = charts.stream().map(p -> p.getFileName().toString()).toList();
        assertEquals(List.of("lookup_gauss_int64_uniform_random.pdf", "speedup_gauss_int64_uniform_random.pdf",
                "p99_gauss_int64_uniform_random.pdf", "model_gauss_int64.pdf", "window_gauss_int64.pdf"), names);
        for (Path chart : charts) {
            String pdf = Files.readString(chart, StandardCharsets.ISO_8859_1);
            assertTrue(pdf.startsWith("%PDF-1.4") && pdf.endsWith("%%EOF"), chart.toString());
            assertTrue(pdf.contains("nn_relu_4x4"), chart + " must have a legend entry for the net");
        }
        assertTrue(Files.readString(charts.getFirst(), StandardCharsets.ISO_8859_1).contains("branchless"));

        String md = Files.readString(dir.resolve(Report.SUMMARY_MD));
        assertTrue(md.contains("## _gauss_int64 / _uniform / random"), md);
        assertTrue(md.contains("| index | 1e5 | 1e6 |"), md);
        assertTrue(md.contains("| binary | 100.0 | 300.0 |"), md);
        assertTrue(md.contains("| branchless | - | 250.0 (x1.20) |"), md);
        assertTrue(md.contains("| nn_relu_4x4 | 50.0 (x2.00) | 100.0 (x3.00) |"), md);
        assertTrue(md.contains("## Learned models"), md);
        assertTrue(md.contains("| _gauss_int64 | 1e5 | nn_relu_4x4 | 8 | 3.0 | 1.50 | 25.0 | 328 |"), md);
        assertTrue(md.contains("allocated 4096 bytes"), md);
        assertTrue(md.contains("fork medians differ by 30.0%"), md);
        assertTrue(md.contains("saw JIT compilation"), md);
        assertTrue(md.contains("(charts/lookup_gauss_int64_uniform_random.pdf)"), md);

        List<String> csv = Files.readAllLines(dir.resolve(Report.SUMMARY_CSV));
        assertEquals(6, csv.size());
        assertTrue(csv.get(0).endsWith("speedup_vs_binary"));
    }

    @Test
    void cleanRunsReportNoWarnings(@TempDir Path dir) throws IOException {
        List<ResultRow> rows = List.of(
                row("_gauss_int64", "_1e5", "_uniform", "random", "binary", 1, 100),
                row("_gauss_int64", "_1e5", "_uniform", "random", "binary", 2, 101));
        List<Path> charts = Report.write(rows, dir, "r.csv");
        String md = Files.readString(dir.resolve(Report.SUMMARY_MD));
        assertTrue(md.contains("## Warnings\n\nNone."), md);
        assertFalse(md.contains("## Learned models"), md);
        assertEquals(3, charts.size(), "a single baseline still gets lookup, speedup and p99 charts");
    }

    @Test
    void chartsWithoutFinitePointsAreSkipped(@TempDir Path dir) throws IOException {
        assertNull(Report.chart(List.of(new Report.Point("a", 5, Double.NaN)), "t", "y", dir.resolve("x.pdf")));
        assertFalse(Files.exists(dir.resolve("x.pdf")));
        assertEquals(dir.resolve("y.pdf"), Report.chart(List.of(new Report.Point("a", 5, -3),
                new Report.Point("a", 6, 2)), "t", "y", dir.resolve("y.pdf")));
    }

    @Test
    void helpersLabelSizesAndOrderIndexes() {
        assertEquals("1e7", Report.sizeLabel("_1e7"));
        assertEquals("custom", Report.sizeLabel("custom"));
        assertEquals(2e8, Report.sizeValue("_max", 5));
        assertEquals(5, Report.sizeValue("custom", 5));
        assertTrue(Report.INDEX_ORDER.compare("binary", "branchless") < 0);
        assertTrue(Report.INDEX_ORDER.compare("branchless", "nn_a") < 0);
        assertTrue(Report.INDEX_ORDER.compare("nn_a", "nn_b") < 0);
        assertThrows(IllegalArgumentException.class, () -> Report.write(List.of(), Path.of("unused"), "r"));
    }
}
