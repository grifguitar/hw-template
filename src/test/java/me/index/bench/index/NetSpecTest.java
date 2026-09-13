package me.index.bench.index;

import me.index.config.parameters.enums.Activation;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NetSpecTest {
    @Test
    void specsParseIntoShapesAndLabels() {
        NetSpec deep = NetSpec.parse(" _leaky_relu : 16x8x4 ");
        assertEquals(Activation._leaky_relu, deep.activation());
        assertEquals(List.of(16, 8, 4), deep.hidden());
        assertArrayEquals(new int[]{1, 16, 8, 4, 1}, deep.layerSizes());
        assertEquals("16x8x4", deep.layersText());
        assertEquals("nn_leaky_relu_16x8x4", deep.label());
        assertEquals("_leaky_relu:16x8x4", deep.toString());

        NetSpec linear = NetSpec.parse("_tanh:none");
        assertArrayEquals(new int[]{1, 1}, linear.layerSizes());
        assertEquals("nn_linear", linear.label());
        assertEquals("none", linear.layersText());

        assertEquals("nn_relu_8", NetSpec.parse("_relu:8").label());
    }

    @Test
    void theTextFormRoundTrips() {
        for (String text : List.of("_relu:4x4", "_sigmoid:none", "_softsign:32")) {
            assertEquals(text, NetSpec.parse(text).toString());
            assertEquals(NetSpec.parse(text), NetSpec.parse(NetSpec.parse(text).toString()));
        }
    }

    @Test
    void listsSkipBlanksAndRejectDuplicates() {
        assertEquals(List.of(), NetSpec.parseList(""));
        assertEquals(2, NetSpec.parseList("_relu:4x4, ,_tanh:8").size());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> NetSpec.parseList("_relu:none,_tanh:none"));
        assertTrue(e.getMessage().contains("nn_linear"), e.getMessage());
    }

    @Test
    void malformedSpecsAreRejectedWithAHelpfulMessage() {
        for (String bad : List.of("_relu", "relu:4", "_relu:", "_relu:4x", "_relu:0", "_relu:-3", "_relu:4,4",
                "_relu:x")) {
            assertThrows(IllegalArgumentException.class, () -> NetSpec.parse(bad), bad);
        }
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> NetSpec.parse("_swish:4"));
        assertTrue(e.getMessage().contains("_softsign"), e.getMessage());
    }

    @Test
    void netLabelsAreRecognised() {
        assertTrue(NetSpec.isNetLabel("nn_relu_4x4"));
        assertTrue(NetSpec.isNetLabel("nn_linear"));
        assertFalse(NetSpec.isNetLabel("binary"));
    }
}
