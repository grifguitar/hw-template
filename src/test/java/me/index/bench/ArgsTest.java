package me.index.bench;

import me.index.config.parameters.enums.Workload;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ArgsTest {
    private static final Set<String> ALLOWED = Set.of("--a", "--n", "--x", "--list", "--path");

    @Test
    void optionsAreReadWithDefaults() {
        Args a = Args.parse(new String[]{"cmd", "--a", "text", "--n", " 12 ", "--x", "0.5", "--list", "p, q,,r",
                "--path", "some/dir"}, 1, ALLOWED);
        assertTrue(a.has("--a"));
        assertEquals("text", a.required("--a"));
        assertEquals(12, a.intValue("--n", 0, 1));
        assertEquals(0.5, a.doubleValue("--x", 1));
        assertEquals(List.of("p", "q", "r"), a.list("--list", ""));
        assertEquals(Path.of("some/dir"), a.path("--path"));

        Args empty = Args.parse(new String[0], 0, ALLOWED);
        assertFalse(empty.has("--a"));
        assertEquals("d", empty.string("--a", "d"));
        assertEquals(3, empty.intValue("--n", 3, 1));
        assertEquals(7L, empty.longValue("--n", 7));
        assertEquals(2.5, empty.doubleValue("--x", 2.5));
        assertEquals(List.of("u", "v"), empty.list("--list", "u,v"));
        assertEquals(List.of(), empty.list("--list", ""));
    }

    @Test
    void malformedCommandLinesAreUsageErrors() {
        assertThrows(Args.UsageException.class, () -> Args.parse(new String[]{"--nope", "1"}, 0, ALLOWED));
        assertThrows(Args.UsageException.class, () -> Args.parse(new String[]{"--a"}, 0, ALLOWED));
        assertThrows(Args.UsageException.class, () -> Args.parse(new String[]{"--a", "1", "--a", "2"}, 0, ALLOWED));

        Args a = Args.parse(new String[]{"--n", "x", "--x", "y"}, 0, ALLOWED);
        assertThrows(Args.UsageException.class, () -> a.required("--a"));
        assertThrows(Args.UsageException.class, () -> a.path("--path"));
        assertThrows(Args.UsageException.class, () -> a.intValue("--n", 0, 0));
        assertThrows(Args.UsageException.class, () -> a.longValue("--n", 0));
        assertThrows(Args.UsageException.class, () -> a.doubleValue("--x", 0));
        assertThrows(Args.UsageException.class,
                () -> Args.parse(new String[]{"--n", "0"}, 0, ALLOWED).intValue("--n", 5, 1));
    }

    @Test
    void enumValuesAreMatchedByNameAndErrorsListTheChoices() {
        assertEquals(Workload._zipf, Args.enumValue("--w", " _zipf ", Workload.values()));
        Args.UsageException e = assertThrows(Args.UsageException.class,
                () -> Args.enumValue("--w", "zipf", Workload.values()));
        assertTrue(e.getMessage().contains("--w") && e.getMessage().contains("_uniform"), e.getMessage());
    }
}
