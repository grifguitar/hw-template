package me.index.bench.report;

import me.index.bench.Stats;
import me.index.config.parameters.enums.DataSize;
import me.index.view.Plot;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.ToDoubleFunction;

/**
 * Aggregates the per-JVM rows (median over forks) into {@code summary.csv}, {@code summary.md} and PDF charts.
 */
public final class Report {
    public static final String SUMMARY_CSV = "summary.csv";
    public static final String SUMMARY_MD = "summary.md";
    public static final String CHARTS = "charts";
    public static final String BINARY = "binary";

    static final double FORK_SPREAD_WARNING = 0.05;
    private static final double MARKER_RADIUS = 3.0;

    static final Comparator<String> INDEX_ORDER =
            Comparator.comparingInt(Report::indexRank).thenComparing(Comparator.naturalOrder());

    private static final Plot.Color[] PALETTE = {
            Plot.BLUE, Plot.ORANGE, Plot.GREEN, Plot.RED, Plot.PURPLE, Plot.BROWN, Plot.GRAY,
            new Plot.Color(0.737, 0.741, 0.133), new Plot.Color(0.090, 0.745, 0.812),
            new Plot.Color(0.890, 0.467, 0.761), new Plot.Color(0.0, 0.0, 0.0), new Plot.Color(0.3, 0.3, 0.8)};

    private Report() {
    }

    /**
     * @param forkSpread (max - min) / median of the per-fork medians
     * @param speedup    median ns of {@code binary} in the same scenario divided by this median, NaN without binary
     */
    public record Summary(String dataset, String size, double sizeValue, int n, String workload, String order,
                          String index, String kind, int forks, double nsMedian, double nsBest, double forkSpread,
                          double blockP99, double modelNs, long window, double meanAbsErr, long indexBytes,
                          long allocBytes, long jitMs, double speedup) {
    }

    record Point(String series, double x, double y) {
    }

    public static List<Path> write(List<ResultRow> rows, Path outDir, String source) throws IOException {
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("no measurements to report");
        }
        Files.createDirectories(outDir);
        List<Summary> summaries = summarize(rows);
        Files.writeString(outDir.resolve(SUMMARY_CSV), summaryCsv(summaries), StandardCharsets.UTF_8);
        List<Path> charts = charts(summaries, outDir.resolve(CHARTS));
        Files.writeString(outDir.resolve(SUMMARY_MD), markdown(summaries, rows, source, outDir, charts),
                StandardCharsets.UTF_8);
        return charts;
    }

    public static List<Summary> summarize(List<ResultRow> rows) {
        Map<List<String>, List<ResultRow>> groups = new LinkedHashMap<>();
        for (ResultRow r : rows) {
            groups.computeIfAbsent(List.of(r.dataset(), r.size(), r.workload(), r.order(), r.index()),
                    k -> new ArrayList<>()).add(r);
        }
        Map<List<String>, Double> binary = new HashMap<>();
        for (Map.Entry<List<String>, List<ResultRow>> e : groups.entrySet()) {
            if (e.getKey().get(4).equals(BINARY)) {
                binary.put(e.getKey().subList(0, 4), Stats.median(column(e.getValue(), ResultRow::nsMedian)));
            }
        }
        List<Summary> out = new ArrayList<>();
        for (Map.Entry<List<String>, List<ResultRow>> e : groups.entrySet()) {
            List<ResultRow> g = e.getValue();
            ResultRow first = g.getFirst();
            double[] medians = column(g, ResultRow::nsMedian);
            double ns = Stats.median(medians);
            double spread = ns > 0 ? (Stats.max(medians) - Stats.min(medians)) / ns : 0;
            Double base = binary.get(e.getKey().subList(0, 4));
            out.add(new Summary(first.dataset(), first.size(), sizeValue(first.size(), first.n()), first.n(),
                    first.workload(), first.order(), first.index(), first.kind(), g.size(),
                    ns, Stats.min(column(g, ResultRow::nsMin)), spread,
                    Stats.median(column(g, ResultRow::blockP99)), finiteMedian(column(g, ResultRow::modelNs)),
                    first.window(), first.meanAbsErr(), first.indexBytes(),
                    g.stream().mapToLong(ResultRow::allocBytes).max().orElse(0),
                    g.stream().mapToLong(ResultRow::jitMs).max().orElse(0),
                    base == null ? Double.NaN : base / ns));
        }
        out.sort(Comparator.comparing(Summary::dataset)
                .thenComparing(Summary::workload)
                .thenComparing(Summary::order)
                .thenComparingDouble(Summary::sizeValue)
                .thenComparing(Summary::index, INDEX_ORDER));
        return out;
    }

    static double sizeValue(String size, int n) {
        for (DataSize d : DataSize.values()) {
            if (d.name().equals(size)) {
                return d.size;
            }
        }
        return n;
    }

    static int indexRank(String name) {
        return switch (name) {
            case BINARY -> 0;
            case "branchless" -> 1;
            default -> 2;
        };
    }

    static String sizeLabel(String size) {
        return size.startsWith("_") ? size.substring(1) : size;
    }

    private static double[] column(List<ResultRow> rows, ToDoubleFunction<ResultRow> f) {
        return rows.stream().mapToDouble(f).toArray();
    }

    private static double finiteMedian(double[] values) {
        double[] finite = Stats.finite(values);
        return finite.length == 0 ? Double.NaN : Stats.median(finite);
    }

    static String summaryCsv(List<Summary> summaries) {
        StringBuilder sb = new StringBuilder("dataset,size,n,workload,order,index,kind,forks,ns_median,ns_best,"
                + "fork_spread,block_p99,model_ns,window,mean_abs_err,index_bytes,alloc_bytes,jit_ms,speedup_vs_binary\n");
        for (Summary s : summaries) {
            sb.append(String.join(",", s.dataset(), s.size(), Integer.toString(s.n()), s.workload(), s.order(),
                    s.index(), s.kind(), Integer.toString(s.forks()), Double.toString(s.nsMedian()),
                    Double.toString(s.nsBest()), Double.toString(s.forkSpread()), Double.toString(s.blockP99()),
                    Double.toString(s.modelNs()), Long.toString(s.window()), Double.toString(s.meanAbsErr()),
                    Long.toString(s.indexBytes()), Long.toString(s.allocBytes()), Long.toString(s.jitMs()),
                    Double.toString(s.speedup()))).append('\n');
        }
        return sb.toString();
    }

    static List<Path> charts(List<Summary> summaries, Path dir) throws IOException {
        Files.createDirectories(dir);
        List<Path> files = new ArrayList<>();
        Map<List<String>, List<Summary>> scenarios = new LinkedHashMap<>();
        Map<String, List<Summary>> nets = new LinkedHashMap<>();
        for (Summary s : summaries) {
            scenarios.computeIfAbsent(List.of(s.dataset(), s.workload(), s.order()), k -> new ArrayList<>()).add(s);
            if (s.kind().equals(ResultRow.NET)) {
                nets.computeIfAbsent(s.dataset(), k -> new ArrayList<>()).add(s);
            }
        }
        for (Map.Entry<List<String>, List<Summary>> e : scenarios.entrySet()) {
            String dataset = e.getKey().get(0);
            String workload = e.getKey().get(1);
            String order = e.getKey().get(2);
            String suffix = dataset + workload + "_" + order + ".pdf";
            String scenario = dataset + " / " + workload + " / " + order;
            List<Summary> list = e.getValue();
            addIfDrawn(files, chart(points(list, Summary::nsMedian), "Lookup time: " + scenario,
                    "ns per lookup (median over forks)", dir.resolve("lookup" + suffix)));
            addIfDrawn(files, chart(points(list, Summary::speedup), "Speedup over binary search: " + scenario,
                    "binary ns / index ns", dir.resolve("speedup" + suffix)));
            addIfDrawn(files, chart(points(list, Summary::blockP99), "p99 lookup time: " + scenario,
                    "ns per lookup, p99 over query blocks", dir.resolve("p99" + suffix)));
        }
        for (Map.Entry<String, List<Summary>> e : nets.entrySet()) {
            Map<List<String>, List<Summary>> perModel = new LinkedHashMap<>();
            for (Summary s : e.getValue()) {
                perModel.computeIfAbsent(List.of(s.index(), s.size()), k -> new ArrayList<>()).add(s);
            }
            List<Point> model = new ArrayList<>();
            List<Point> window = new ArrayList<>();
            for (List<Summary> same : perModel.values()) {
                Summary s = same.getFirst();
                double x = Math.log10(s.sizeValue());
                model.add(new Point(s.index(), x,
                        finiteMedian(same.stream().mapToDouble(Summary::modelNs).toArray())));
                window.add(new Point(s.index(), x, Math.log(s.window()) / Math.log(2)));
            }
            addIfDrawn(files, chart(model, "Model inference time: " + e.getKey(), "ns per prediction",
                    dir.resolve("model" + e.getKey() + ".pdf")));
            addIfDrawn(files, chart(window, "Last-mile search window: " + e.getKey(), "log2(window size)",
                    dir.resolve("window" + e.getKey() + ".pdf")));
        }
        return files;
    }

    private static void addIfDrawn(List<Path> files, Path chart) {
        if (chart != null) {
            files.add(chart);
        }
    }

    private static List<Point> points(List<Summary> list, ToDoubleFunction<Summary> y) {
        return list.stream().map(s -> new Point(s.index(), Math.log10(s.sizeValue()), y.applyAsDouble(s))).toList();
    }

    static Path chart(List<Point> points, String title, String yLabel, Path file) throws IOException {
        Map<String, List<Point>> series = new TreeMap<>(INDEX_ORDER);
        for (Point p : points) {
            if (Double.isFinite(p.x()) && Double.isFinite(p.y())) {
                series.computeIfAbsent(p.series(), k -> new ArrayList<>()).add(p);
            }
        }
        if (series.isEmpty()) {
            return null;
        }
        Plot plot = new Plot(900, 600).title(title).xlabel("log10(number of keys)").ylabel(yLabel);
        double minX = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY;
        double maxY = Double.NEGATIVE_INFINITY;
        int color = 0;
        for (Map.Entry<String, List<Point>> e : series.entrySet()) {
            List<Point> ps = new ArrayList<>(e.getValue());
            ps.sort(Comparator.comparingDouble(Point::x));
            double[] xs = ps.stream().mapToDouble(Point::x).toArray();
            double[] ys = ps.stream().mapToDouble(Point::y).toArray();
            Plot.Color c = PALETTE[color++ % PALETTE.length];
            plot.line(xs, ys, c, e.getKey()).scatter(xs, ys, c, "", MARKER_RADIUS);
            minX = Math.min(minX, xs[0]);
            maxX = Math.max(maxX, xs[xs.length - 1]);
            for (double y : ys) {
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
        }
        plot.xlim(minX - 0.5, maxX + 0.5).xTicks((int) Math.round((maxX - minX + 1) * 2));
        if (minY >= 0 && maxY > 0) {
            plot.ylim(0, maxY * 1.1);
        }
        plot.save(file.toString());
        return file;
    }

    static String markdown(List<Summary> summaries, List<ResultRow> rows, String source, Path outDir,
                           List<Path> charts) {
        StringBuilder md = new StringBuilder("# Learned index benchmark\n\n");
        int maxForks = summaries.stream().mapToInt(Summary::forks).max().orElse(0);
        md.append(String.format(Locale.ROOT, "Source: `%s`, %d measurement(s), up to %d fork(s) per configuration.%n%n",
                source, rows.size(), maxForks));
        md.append("Cells hold the median over forks of the per-JVM median time per lookup in ns; ")
                .append("`x` is the speedup over `binary` in the same scenario.\n\n");

        Map<List<String>, List<Summary>> scenarios = new LinkedHashMap<>();
        for (Summary s : summaries) {
            scenarios.computeIfAbsent(List.of(s.dataset(), s.workload(), s.order()), k -> new ArrayList<>()).add(s);
        }
        for (Map.Entry<List<String>, List<Summary>> e : scenarios.entrySet()) {
            md.append("## ").append(String.join(" / ", e.getKey())).append("\n\n");
            List<Summary> list = e.getValue();
            TreeMap<Double, String> sizes = new TreeMap<>();
            TreeSet<String> indexes = new TreeSet<>(INDEX_ORDER);
            Map<List<Object>, Summary> cell = new HashMap<>();
            for (Summary s : list) {
                sizes.put(s.sizeValue(), s.size());
                indexes.add(s.index());
                cell.put(List.of(s.index(), s.sizeValue()), s);
            }
            md.append("| index |");
            sizes.values().forEach(size -> md.append(' ').append(sizeLabel(size)).append(" |"));
            md.append("\n|---|");
            sizes.values().forEach(size -> md.append("---:|"));
            md.append('\n');
            for (String index : indexes) {
                md.append("| ").append(index).append(" |");
                for (double size : sizes.keySet()) {
                    Summary s = cell.get(List.of(index, size));
                    if (s == null) {
                        md.append(" - |");
                        continue;
                    }
                    md.append(String.format(Locale.ROOT, " %.1f", s.nsMedian()));
                    if (!index.equals(BINARY) && Double.isFinite(s.speedup())) {
                        md.append(String.format(Locale.ROOT, " (x%.2f)", s.speedup()));
                    }
                    md.append(" |");
                }
                md.append('\n');
            }
            md.append('\n');
        }

        List<Summary> nets = summaries.stream().filter(s -> s.kind().equals(ResultRow.NET)).toList();
        if (!nets.isEmpty()) {
            md.append("## Learned models\n\n")
                    .append("| dataset | size | index | window | log2(window) | mean abs err | model ns | model bytes |\n")
                    .append("|---|---|---|---:|---:|---:|---:|---:|\n");
            Map<List<String>, List<Summary>> perModel = new LinkedHashMap<>();
            for (Summary s : nets) {
                perModel.computeIfAbsent(List.of(s.dataset(), s.size(), s.index()), k -> new ArrayList<>()).add(s);
            }
            List<List<Summary>> ordered = new ArrayList<>(perModel.values());
            ordered.sort(Comparator.<List<Summary>, String>comparing(g -> g.getFirst().dataset())
                    .thenComparingDouble(g -> g.getFirst().sizeValue())
                    .thenComparing(g -> g.getFirst().index(), INDEX_ORDER));
            for (List<Summary> g : ordered) {
                Summary s = g.getFirst();
                md.append(String.format(Locale.ROOT, "| %s | %s | %s | %d | %.1f | %.2f | %.1f | %d |%n",
                        s.dataset(), sizeLabel(s.size()), s.index(), s.window(), Math.log(s.window()) / Math.log(2),
                        s.meanAbsErr(), finiteMedian(g.stream().mapToDouble(Summary::modelNs).toArray()),
                        s.indexBytes()));
            }
            md.append('\n');
        }

        md.append("## Warnings\n\n");
        List<String> warnings = new ArrayList<>();
        for (ResultRow r : rows) {
            String where = r.dataset() + " / " + sizeLabel(r.size()) + " / " + r.workload() + " / " + r.order()
                    + " / " + r.index() + " fork " + r.fork();
            if (r.allocBytes() > 0) {
                warnings.add(where + ": the measured passes allocated " + r.allocBytes() + " bytes"
                        + (r.jitMs() > 0 ? " while the JIT was still compiling; raise WARMUP_MIN" : ""));
            } else if (r.allocBytes() < 0) {
                warnings.add(where + ": this JVM cannot track allocations");
            }
        }
        for (Summary s : summaries) {
            if (s.forks() > 1 && s.forkSpread() > FORK_SPREAD_WARNING) {
                warnings.add(String.format(Locale.ROOT, "%s / %s / %s / %s / %s: fork medians differ by %.1f%%",
                        s.dataset(), sizeLabel(s.size()), s.workload(), s.order(), s.index(), 100 * s.forkSpread()));
            }
        }
        long jitRows = rows.stream().filter(r -> r.jitMs() > 0).count();
        if (jitRows > 0) {
            warnings.add(String.format(Locale.ROOT,
                    "%d of %d measurement(s) saw JIT compilation during the timed passes (max %d ms);"
                            + " raise WARMUP_MIN if this is large", jitRows, rows.size(),
                    rows.stream().mapToLong(ResultRow::jitMs).max().orElse(0)));
        }
        if (warnings.isEmpty()) {
            md.append("None.\n");
        } else {
            warnings.forEach(w -> md.append("- ").append(w).append('\n'));
        }

        if (!charts.isEmpty()) {
            md.append("\n## Charts\n\n");
            for (Path chart : charts) {
                String relative = outDir.toAbsolutePath().relativize(chart.toAbsolutePath()).toString()
                        .replace('\\', '/');
                md.append("- [").append(chart.getFileName()).append("](").append(relative).append(")\n");
            }
        }
        return md.toString();
    }
}
