package me.index.config.parameters.records;

import me.index.config.Config;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Properties;

public record HiddenLayers(List<Integer> sizes) {
    public static final String KEY = "net.hidden.layers";
    public static final HiddenLayers DEFAULT = new HiddenLayers(List.of(4, 4));

    public HiddenLayers {
        sizes = List.copyOf(sizes);
        for (int size : sizes) {
            if (size <= 0) {
                throw new IllegalArgumentException("property '" + KEY + "': sizes must be positive, got " + size);
            }
        }
    }

    public static Optional<HiddenLayers> parse(Properties p) {
        return Config.parseRecord(p, KEY, "comma-separated ints", HiddenLayers::parse);
    }

    private static HiddenLayers parse(String text) {
        if (text.isEmpty()) {
            return new HiddenLayers(List.of());
        }
        List<Integer> sizes = new ArrayList<>();
        for (String part : text.split(",", -1)) {
            sizes.add(Integer.parseInt(part.trim()));
        }
        return new HiddenLayers(sizes);
    }

    public int[] layerSizes() {
        int[] layers = new int[sizes.size() + 2];
        layers[0] = 1;
        for (int i = 0; i < sizes.size(); i++) {
            layers[i + 1] = sizes.get(i);
        }
        layers[layers.length - 1] = 1;
        return layers;
    }
}
