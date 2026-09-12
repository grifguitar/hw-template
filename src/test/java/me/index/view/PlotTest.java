package me.index.view;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class PlotTest {
    private static String render(Plot plot, Path dir, String name) throws IOException {
        Path out = dir.resolve(name);
        plot.save(out.toString());
        return Files.readString(out, StandardCharsets.ISO_8859_1);
    }

    @Test
    void producesAWellFormedPdf(@TempDir Path dir) throws IOException {
        String pdf = render(new Plot(400, 300)
                .title("t").xlabel("x").ylabel("y")
                .scatter(new double[]{0, 1, 2}, new double[]{0, 1, 4}, Plot.BLUE, "s"), dir, "a.pdf");

        assertTrue(pdf.startsWith("%PDF-1.4"), "missing header");
        assertTrue(pdf.endsWith("%%EOF"), "missing trailer");
        assertTrue(pdf.contains("/Type /Catalog"));
        assertTrue(pdf.contains("/Type /Pages /Count 1"));
        assertTrue(pdf.contains("/MediaBox [0 0 400 300]"));
        assertTrue(pdf.contains("/BaseFont /Helvetica"));

        int xrefAt = pdf.lastIndexOf("\nxref\n") + 1;
        int startXref = Integer.parseInt(pdf.substring(pdf.lastIndexOf("startxref\n") + 10, pdf.length() - 6).trim());
        assertEquals(xrefAt, startXref, "startxref must point at the xref table");
        String[] rows = pdf.substring(xrefAt).split("\n");
        for (int obj = 1; obj + 2 < rows.length && rows[obj + 2].endsWith(" n "); obj++) {
            int offset = Integer.parseInt(rows[obj + 2].substring(0, 10));
            assertTrue(pdf.startsWith(obj + " 0 obj", offset),
                    "xref entry " + obj + " points at " + offset);
        }
    }

    @Test
    void dataIsClippedToThePlotArea(@TempDir Path dir) throws IOException {
        String pdf = render(new Plot(400, 300)
                .xlim(0, 1)
                .scatter(new double[]{-100, 0.5, 100}, new double[]{0, 0.5, 1}, Plot.RED, "s"), dir, "b.pdf");
        assertTrue(pdf.contains("W n"), "scatter must be wrapped in a clipping path");
    }

    @Test
    void degenerateRangesDoNotBlowUp(@TempDir Path dir) {
        assertDoesNotThrow(() -> render(new Plot(300, 200), dir, "empty.pdf"));
        assertDoesNotThrow(() -> render(new Plot(300, 200)
                .scatter(new double[]{1, 1, 1}, new double[]{2, 2, 2}, Plot.BLUE, "flat"), dir, "flat.pdf"));
        assertDoesNotThrow(() -> render(new Plot(300, 200)
                .xlim(5, 5).ylim(5, 5)
                .line(new double[]{5}, new double[]{5}, Plot.GREEN, "dot"), dir, "point.pdf"));
    }

    @Test
    void invertedLimitsAreReported(@TempDir Path dir) {
        assertThrows(IllegalStateException.class, () -> render(new Plot(300, 200)
                .xlim(10, 0)
                .scatter(new double[]{1}, new double[]{1}, Plot.BLUE, "s"), dir, "inv.pdf"));
    }

    @Test
    void nonFiniteDataDoesNotDestroyTheAxes(@TempDir Path dir) {
        assertDoesNotThrow(() -> render(new Plot(300, 200)
                .scatter(new double[]{0, 1, Double.NaN}, new double[]{0, 1, Double.POSITIVE_INFINITY},
                        Plot.BLUE, "s"), dir, "nan.pdf"));
    }

    @Test
    void pageNumberFormattingIsLocaleIndependent() {
        assertEquals("5", Page.fmt(5.0));
        assertEquals("-3", Page.fmt(-3.0));
        assertEquals("1.25", Page.fmt(1.25));
        assertFalse(Page.fmt(1.25).contains(","), "decimal separator must be a dot");
    }

    @Test
    void escapeProducesValidWinAnsiLiterals() {
        assertEquals("plain text", Page.escape("plain text"));
        assertEquals("a\\(b\\)c\\\\d", Page.escape("a(b)c\\d"));
        assertEquals("???", Page.escape("абв"));
        assertEquals("\\200", Page.escape("€"));
        assertEquals("\\227", Page.escape("—"));
        assertEquals("\\351", Page.escape("é"));
        assertEquals("?", Page.escape("\n"));
        for (char c : Page.escape("€—éА").toCharArray()) {
            assertTrue(c < 128, "escape() must emit pure ASCII, got 0x" + Integer.toHexString(c));
        }
    }

    @Test
    void saveIsIdempotent(@TempDir Path dir) throws IOException {
        Plot plot = new Plot(300, 200).title("t")
                .scatter(new double[]{0, 1}, new double[]{0, 1}, Plot.BLUE, "s");
        String first = render(plot, dir, "once.pdf");
        String second = render(plot, dir, "twice.pdf");
        assertEquals(first, second, "a second save() must not append the drawing again");
    }

    @Test
    void tickCountsMustBePositive() {
        Plot plot = new Plot(300, 200);
        assertThrows(IllegalArgumentException.class, () -> plot.xTicks(0));
        assertThrows(IllegalArgumentException.class, () -> plot.yTicks(0));
        assertThrows(IllegalArgumentException.class, () -> plot.xTicks(-3));
        assertThrows(IllegalArgumentException.class, () -> plot.yTicks(-1));
        assertDoesNotThrow(() -> plot.xTicks(1).yTicks(1));
    }

    @Test
    void nonFinitePointsAreSkippedAndNeverReachThePdf(@TempDir Path dir) throws IOException {
        String pdf = render(new Plot(400, 300)
                .scatter(new double[]{0, Double.NaN, 1}, new double[]{0, 0.5, 1}, Plot.BLUE, "s")
                .line(new double[]{0, 0.5, 1}, new double[]{0, Double.POSITIVE_INFINITY, 1}, Plot.RED, "l"),
                dir, "skip.pdf");
        assertFalse(pdf.contains("NaN"), "NaN must never be written into a PDF");
        assertFalse(pdf.contains("Infinity"), "Infinity must never be written into a PDF");
    }

    @Test
    void allSeriesOverloadsProduceDrawableLayers(@TempDir Path dir) throws IOException {
        long[] lx = {0, 1, 2};
        long[] ly = {0, 2, 4};
        String pdf = render(new Plot(500, 400)
                .grid(false).legend(true).xTicks(3).yTicks(2).squareMarkers(true)
                .line(new double[]{0, 1}, new double[]{0, 1}, "auto-color line")
                .line(lx, ly, Plot.GREEN, "long line", 2.0)
                .line(lx, ly, Plot.PURPLE, "long line 2")
                .line(lx, ly, "long line 3")
                .scatter(new double[]{0, 1}, new double[]{1, 0}, "auto-color scatter")
                .scatter(lx, ly, Plot.BROWN, "long scatter", 1.5)
                .scatter(lx, ly, Plot.GRAY, "long scatter 2")
                .scatter(lx, ly, "long scatter 3"),
                dir, "overloads.pdf");
        assertTrue(pdf.startsWith("%PDF-1.4"));
        assertTrue(pdf.endsWith("%%EOF"));
        for (String label : new String[]{"auto-color line", "long line 3", "long scatter 3"}) {
            assertTrue(pdf.contains(label), "legend must list " + label);
        }
    }

    @Test
    void gridCanBeTurnedOff(@TempDir Path dir) throws IOException {
        String withGrid = render(new Plot(300, 200)
                .scatter(new double[]{0, 1}, new double[]{0, 1}, Plot.BLUE, "s"), dir, "grid.pdf");
        String noGrid = render(new Plot(300, 200).grid(false)
                .scatter(new double[]{0, 1}, new double[]{0, 1}, Plot.BLUE, "s"), dir, "nogrid.pdf");
        assertTrue(noGrid.length() < withGrid.length(), "disabling the grid must emit fewer operators");
    }

    @Test
    void legendCanBeSuppressed(@TempDir Path dir) throws IOException {
        String hidden = render(new Plot(300, 200).legend(false)
                .scatter(new double[]{0, 1}, new double[]{0, 1}, Plot.BLUE, "secret"), dir, "nolegend.pdf");
        assertFalse(hidden.contains("secret"), "legend(false) must hide the label");
    }

    @Test
    void pageDrawingPrimitivesEmitTheRightOperators(@TempDir Path dir) throws IOException {
        Document doc = new Document();
        Page page = doc.addPage(100, 100);
        page.square(50, 50, 10).fillAndStroke();
        page.dash(3, 2).moveTo(0, 0).lineTo(10, 10).stroke().solidLine();
        Path out = dir.resolve("prims.pdf");
        doc.save(out.toString());
        String pdf = Files.readString(out, StandardCharsets.ISO_8859_1);
        assertTrue(pdf.contains("40 40 20 20 re"), "square() must emit a centred rect");
        assertTrue(pdf.contains("\nB\n"), "fillAndStroke() must emit B");
        assertTrue(pdf.contains("[3 2 ] 0 d"), "dash() must emit the pattern");
        assertTrue(pdf.contains("[] 0 d"), "solidLine() must reset the pattern");
    }

    @Test
    void fmtRejectsNonFiniteNumbers() {
        assertThrows(IllegalArgumentException.class, () -> Page.fmt(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Page.fmt(Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> Page.fmt(Double.NEGATIVE_INFINITY));
    }
}
