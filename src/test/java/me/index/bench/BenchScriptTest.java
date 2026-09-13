package me.index.bench;

import me.index.bench.data.QueryOrder;
import me.index.bench.index.NetSpec;
import me.index.config.parameters.enums.DataSize;
import me.index.config.parameters.enums.Keyset;
import me.index.config.parameters.enums.LossFunction;
import me.index.config.parameters.enums.Workload;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class BenchScriptTest {
    private static final Path SCRIPT = Path.of("bench.sh");

    private static String script() throws IOException {
        return Files.readString(SCRIPT, StandardCharsets.UTF_8);
    }

    private record Result(int code, String output) {
    }

    private static boolean bashAvailable() {
        try {
            return new ProcessBuilder("bash", "-c", "true").start().waitFor() == 0;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private static Result bash(String... args) throws IOException, InterruptedException {
        List<String> command = new java.util.ArrayList<>(List.of("bash"));
        command.addAll(List.of(args));
        Process p = new ProcessBuilder(command).redirectErrorStream(true).start();
        String output = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertTrue(p.waitFor(30, TimeUnit.SECONDS), "bash did not finish");
        return new Result(p.exitValue(), output);
    }

    @Test
    void theScriptIsPresentAndExecutable() {
        assertTrue(Files.isRegularFile(SCRIPT), "bench.sh must live at the repo root");
        assertTrue(Files.isExecutable(SCRIPT), "bench.sh must be executable");
    }

    @Test
    void theScriptParses() throws Exception {
        assumeTrue(bashAvailable(), "bash is not installed");
        Result r = bash("-n", SCRIPT.toString());
        assertEquals(0, r.code(), r.output());
    }

    @Test
    void helpPrintsTheSettingsAndArgumentsAreRejected() throws Exception {
        assumeTrue(bashAvailable(), "bash is not installed");
        Result help = bash(SCRIPT.toString(), "--help");
        assertEquals(0, help.code(), help.output());
        assertTrue(help.output().contains("PROFILE") && help.output().contains("WORKLOADS"), help.output());
        assertTrue(!help.output().contains("set -euo"), "help must stop at the end of the header");

        assertEquals(2, bash(SCRIPT.toString(), "unexpected").code());
    }

    @Test
    void everyOptionTheScriptPassesIsKnownToBenchMain() throws IOException {
        Set<String> allowed = new HashSet<>(BenchMain.allOptions());
        allowed.addAll(Set.of("--help", "--cpunodebind", "--membind"));
        Matcher m = Pattern.compile("(?<![\\w-])--[a-z][a-z-]*").matcher(script());
        int found = 0;
        while (m.find()) {
            assertTrue(allowed.contains(m.group()), "bench.sh uses unknown option " + m.group());
            found++;
        }
        assertTrue(found > 20, "the option scan found too little: " + found);
        for (String option : BenchMain.RUN_OPTIONS) {
            assertTrue(script().contains(option + " "), "bench.sh never passes " + option);
        }
    }

    @Test
    void theScriptUsesTheFileNamesThatPrepareWrites() throws IOException {
        String text = script();
        assertTrue(text.contains("queries${wl}_${ord}.bin"));
        assertEquals("queries_uniform_random.bin", Prepare.queriesFileName(Workload._uniform, QueryOrder.random));
        assertTrue(text.contains(Prepare.KEYS_FILE));
        assertTrue(text.contains("MODELS_LIST=" + Prepare.MODELS_LIST));
        assertTrue(text.contains("$idx.model"));
        assertEquals("nn_relu_4x4.model", Prepare.modelFileName(NetSpec.parse("_relu:4x4")));
        assertTrue(text.contains("me.index.bench.BenchMain prepare"));
        assertTrue(text.contains("me.index.bench.BenchMain run"));
        assertTrue(text.contains("me.index.bench.BenchMain report"));
        assertTrue(text.contains("-XX:+UseEpsilonGC") && text.contains("-XX:+AlwaysPreTouch"));
    }

    @Test
    void everyDefaultInTheScriptNamesSomethingThatExists() throws IOException {
        Matcher m = Pattern.compile("\\$\\{([A-Z_]+):?=([^}]*)}").matcher(script());
        Set<String> checked = new HashSet<>();
        while (m.find()) {
            String variable = m.group(1);
            for (String token : m.group(2).trim().split("\\s+")) {
                if (token.isEmpty()) {
                    continue;
                }
                switch (variable) {
                    case "DATASETS" -> assertDoesNotThrow(() -> Keyset.valueOf(token), token);
                    case "SIZES" -> assertDoesNotThrow(() -> DataSize.valueOf(token), token);
                    case "WORKLOADS" -> assertDoesNotThrow(() -> Workload.valueOf(token), token);
                    case "ORDERS" -> assertDoesNotThrow(() -> QueryOrder.valueOf(token), token);
                    case "NETS" -> assertDoesNotThrow(() -> NetSpec.parse(token), token);
                    case "BASELINES" -> assertTrue(BenchMain.BASELINES.contains(token), token);
                    case "LOSS" -> assertDoesNotThrow(() -> LossFunction.valueOf(token), token);
                    default -> {
                        continue;
                    }
                }
                checked.add(variable);
            }
        }
        assertEquals(Set.of("DATASETS", "SIZES", "WORKLOADS", "ORDERS", "NETS", "BASELINES", "LOSS"), checked);
        for (String sosd : List.of("wiki_ts_200M_uint64", "books_200M_uint32", "books_800M_uint64",
                "osm_cellids_800M_uint64", "fb_200M_uint64")) {
            assertTrue(script().contains(sosd), "bench.sh must look for " + sosd);
            assertDoesNotThrow(() -> Keyset.valueOf("_" + sosd));
        }
    }
}
