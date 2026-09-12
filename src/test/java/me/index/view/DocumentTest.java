package me.index.view;

import me.index.io.CntOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class DocumentTest {

    private static String save(Document doc, Path dir, String name) throws IOException {
        Path out = dir.resolve(name);
        doc.save(out.toString());
        return Files.readString(out, StandardCharsets.ISO_8859_1);
    }

    @Test
    void a4PageSizesAreCorrect(@TempDir Path dir) throws IOException {
        Document portrait = new Document();
        portrait.addA4VerticalPage().text("x", 10, 10, Font.TIMES_ROMAN, 10);
        assertTrue(save(portrait, dir, "p.pdf").contains("/MediaBox [0 0 595 842]"));

        Document landscape = new Document();
        landscape.addA4HorizontalPage().text("x", 10, 10, Font.COURIER, 10);
        assertTrue(save(landscape, dir, "l.pdf").contains("/MediaBox [0 0 842 595]"));
    }

    @Test
    void emptyDocumentIsRejectedInsteadOfWritingAPageLessPdf(@TempDir Path dir) {
        Document doc = new Document();
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> doc.save(dir.resolve("empty.pdf").toString()));
        assertTrue(e.getMessage().contains("without pages"), e.getMessage());
    }

    @Test
    void everyRegisteredFontGetsItsOwnResourceEntry(@TempDir Path dir) throws IOException {
        Document doc = new Document();
        Page p = doc.addPage(200, 100);
        p.text("a", 10, 10, Font.HELVETICA, 10);
        p.text("b", 10, 30, Font.HELVETICA_BOLD, 10);
        p.text("c", 10, 50, Font.HELVETICA, 10);
        String pdf = save(doc, dir, "fonts.pdf");

        assertTrue(pdf.contains("/BaseFont /Helvetica "), pdf);
        assertTrue(pdf.contains("/BaseFont /Helvetica-Bold"), pdf);
        assertTrue(pdf.contains("/F0 "), "first font must be F0");
        assertTrue(pdf.contains("/F1 "), "second font must be F1");
        assertEquals(1, pdf.split("/BaseFont /Helvetica ", -1).length - 1, "font must not be emitted twice");
    }

    @Test
    void unregisteredFontIsReported() {
        Document doc = new Document();
        assertThrows(IllegalStateException.class, () -> doc.getRegFontName(Font.TIMES_BOLD));
    }

    @Test
    void multiplePagesAreAllWritten(@TempDir Path dir) throws IOException {
        Document doc = new Document();
        doc.addPage(100, 100).text("one", 5, 5, Font.HELVETICA, 8);
        doc.addPage(120, 140).text("two", 5, 5, Font.HELVETICA, 8);
        String pdf = save(doc, dir, "multi.pdf");
        assertTrue(pdf.contains("/Count 2"));
        assertTrue(pdf.contains("/MediaBox [0 0 100 100]"));
        assertTrue(pdf.contains("/MediaBox [0 0 120 140]"));
    }

    @Test
    void countingStreamTracksEveryWritePath() throws IOException {
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        try (CntOutputStream out = new CntOutputStream(sink)) {
            out.write('a');
            out.write(new byte[]{1, 2, 3});
            out.write(new byte[]{4, 5, 6, 7}, 1, 2);
            out.write("hello");
            assertEquals(1 + 3 + 2 + 5, out.count());
        }
        assertEquals(11, sink.size());
    }
}
