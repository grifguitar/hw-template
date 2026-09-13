package me.index.bench.index;

import me.index.config.parameters.enums.Activation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A net to benchmark, written as {@code <activation>:<hidden layers>}: {@code _relu:4x4}, {@code _tanh:8},
 * or {@code _relu:none} for a net without hidden layers (a linear model).
 */
public record NetSpec(Activation activation, List<Integer> hidden) {
    public static final String PREFIX = "nn_";
    public static final String NONE = "none";

    public NetSpec {
        Objects.requireNonNull(activation, "activation");
        hidden = List.copyOf(hidden);
        for (int size : hidden) {
            if (size <= 0) {
                throw new IllegalArgumentException("hidden layer sizes must be positive, got " + hidden);
            }
        }
    }

    public static NetSpec parse(String text) {
        String t = text.trim();
        int colon = t.indexOf(':');
        if (colon < 0) {
            throw new IllegalArgumentException("net spec must look like <activation>:<layers>"
                    + " (e.g. _relu:4x4 or _relu:none), got '" + text + "'");
        }
        String name = t.substring(0, colon).trim();
        String layers = t.substring(colon + 1).trim();
        Activation activation = Arrays.stream(Activation.values())
                .filter(a -> a.name().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown activation '" + name + "' in net spec '"
                        + text + "'; allowed values: " + Arrays.toString(Activation.values())));
        if (layers.equals(NONE)) {
            return new NetSpec(activation, List.of());
        }
        List<Integer> sizes = new ArrayList<>();
        for (String part : layers.split("x", -1)) {
            try {
                sizes.add(Integer.parseInt(part));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("hidden layers in net spec '" + text
                        + "' must be sizes joined by 'x' or '" + NONE + "'", e);
            }
        }
        return new NetSpec(activation, sizes);
    }

    public static List<NetSpec> parseList(String text) {
        List<NetSpec> specs = new ArrayList<>();
        Set<String> labels = new HashSet<>();
        for (String part : text.split(",")) {
            if (part.isBlank()) {
                continue;
            }
            NetSpec spec = parse(part);
            if (!labels.add(spec.label())) {
                throw new IllegalArgumentException("net " + spec.label() + " is listed twice");
            }
            specs.add(spec);
        }
        return specs;
    }

    public static boolean isNetLabel(String name) {
        return name.startsWith(PREFIX);
    }

    public int[] layerSizes() {
        int[] layers = new int[hidden.size() + 2];
        layers[0] = 1;
        for (int i = 0; i < hidden.size(); i++) {
            layers[i + 1] = hidden.get(i);
        }
        layers[layers.length - 1] = 1;
        return layers;
    }

    public String layersText() {
        return hidden.isEmpty() ? NONE : hidden.stream().map(String::valueOf).collect(Collectors.joining("x"));
    }

    public String label() {
        return hidden.isEmpty() ? PREFIX + "linear" : PREFIX + activation.name().substring(1) + "_" + layersText();
    }

    @Override
    public String toString() {
        return activation.name() + ":" + layersText();
    }
}
