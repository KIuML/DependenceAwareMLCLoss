package de.lmu.dal.postprocessing;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionOneLayerPartition;
import ai.libs.jaicore.basic.kvstore.KVStoreSequentialComparator;

public class PairWiseLossCurves {

	private static final List<String> DATASETS = Arrays.asList("birds", "emotions", "enron-f", "flags", "genbase", "llog-f", "medical", "scene", "yeast");
	private static final List<String> ALGOS = Arrays.asList("cc", "lc", "br", "clus", "clus-ensemble", "rakel-2", "rakel-3", "rakel-4", "rakel-5", "boomer_label_wise_logistic_loss", "boomer_example_wise_logistic_loss");
	private static final boolean BINOMIAL = false;

	private static final String FAMILY_TO_PLOT = BINOMIAL ? "moebius" : "polynomial";

	public static void main(final String[] args) throws IOException {
		Map<String, String> replaceNames = new HashMap<>();
		replaceNames.put("cc", "CC");
		replaceNames.put("clus-ensemble", "EPCT");
		replaceNames.put("clus", "PCT");
		replaceNames.put("br", "BR");
		replaceNames.put("lc", "LP");
		replaceNames.put("rakel-2", "RAkEL-2");
		replaceNames.put("rakel-3", "RAkEL-3");
		replaceNames.put("rakel-4", "RAkEL-4");
		replaceNames.put("rakel-5", "RAkEL-5");
		replaceNames.put("boomer_label_wise_logistic_loss", "BOOMER-1");
		replaceNames.put("boomer_example_wise_logistic_loss", "BOOMER-K");

		KVStoreCollection col = new KVStoreCollection(FileUtils.readFileToString(new File("results.kvstore")));
		col.addAll(new KVStoreCollection(FileUtils.readFileToString(new File("results_boomer_v3.kvstore"))));
		KVStoreCollection filtered = new KVStoreCollection();
		col.stream().filter(x -> ALGOS.contains(x.getAsString("algorithm")) && DATASETS.contains(x.getAsString("dataset"))).forEach(filtered::add);
		filtered.stream().forEach(x -> x.put("algorithm", replaceNames.get(x.getAsString("algorithm"))));

		KVStoreCollection plots = new KVStoreCollection();
		plots.setCollectionID("plots");

		KVStoreCollectionOneLayerPartition partition = new KVStoreCollectionOneLayerPartition("dataset", filtered);
		for (Entry<String, KVStoreCollection> part : partition) {
			col = part.getValue();
			col.sort(new KVStoreSequentialComparator("algorithm"));
			for (int i = 0; i < col.size() - 1; i++) {
				IKVStore one = col.get(i);
				for (int j = i + 1; j < col.size(); j++) {
					IKVStore other = col.get(j);

					IKVStore comparison = new KVStore(one.toString());

					comparison.put("moebius", combine(one.getAsStringList("moebius"), other.getAsStringList("moebius")));
					comparison.put("polynomial", combine(one.getAsStringList("polynomial"), other.getAsStringList("polynomial")));
					comparison.put("x", one.getAsString("algorithm"));
					comparison.put("y", other.getAsString("algorithm") + "/" + one.getAsString("algorithm"));
					comparison.put("algorithm", one.getAsString("algorithm") + "-" + other.getAsString("algorithm"));

					plots.add(comparison);
				}
			}
		}

		KVStoreCollectionOneLayerPartition comparisonPartition = new KVStoreCollectionOneLayerPartition("algorithm", plots);
		StringBuilder sb = new StringBuilder();

		for (Entry<String, KVStoreCollection> part : comparisonPartition) {
			col = part.getValue();
			col.sort(new KVStoreSequentialComparator("dataset"));

			// output the plots as a tickz picture
			sb.append("\\begin{my}\n");
			sb.append("\\begin{tikzpicture}\n").append(
					"\\begin{axis}[xmin=1,xmax=1000,xmode=log,ymin=.5,ymax=1.5,legend pos=outer north east,extra x ticks={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9},extra y ticks={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5}, tick style={grid=major},xlabel=")
					.append((BINOMIAL ? "$k/K$" : "$\\alpha$")).append(",ylabel=").append(col.get(0).getAsString("y")).append("]\n");
			for (int i = 0; i < col.size(); i++) {
				sb.append(storeToPlot(col.get(i), "color" + (i + 1)));
			}
			sb.append("\\addplot[mark=none,draw=black] coordinates {\n(1.0,1.0)\n(1000.0,1.0)\n};\n");
			sb.append("\\legend{};");
			sb.append("\\end{axis}\n").append("\\end{tikzpicture}\n").append("\\end{my}\n\n");
		}

		File f = new File("comparison-plots-byk/all-" + FAMILY_TO_PLOT + ".tex");
		f.getParentFile().mkdirs();
		System.out.println(f.getCanonicalPath());
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(f))) {
			bw.write(frameDoc(sb.toString()));
			// bw.write(sb.toString());
		}
	}

	private static String frameDoc(final String docString) {
		StringBuilder sb = new StringBuilder();

		sb.append("\\documentclass[multi=my,crop]{standalone}\n");

		sb.append("\\usepackage[utf8]{inputenc}\n");
		Arrays.asList("graphicx", "tikz", "pgfplots").stream().map(x -> "\\usepackage{" + x + "}\n").forEach(sb::append);

		Map<String, String> colors = new HashMap<>();
		colors.put("color1", "023EFF");
		colors.put("color2", "FF7C00");
		colors.put("color3", "1AC938");
		colors.put("color4", "E8000B");
		colors.put("color5", "8B2BE2");
		colors.put("color6", "9F4800");
		colors.put("color7", "F14CC1");
		colors.put("color8", "A3A3A3");
		colors.put("color9", "00D7FF");
		for (Entry<String, String> color : colors.entrySet()) {
			sb.append("\\definecolor{" + color.getKey() + "}{HTML}{" + color.getValue() + "}\n");
		}
		sb.append("\\pgfplotsset{every axis plot/.append style={thick}}\n");

		sb.append("\n\\begin{document}\n\n");
		sb.append(docString).append("\n");
		sb.append("\n\\end{document}\n");

		return sb.toString();
	}

	private static String storeToPlot(final IKVStore store, final String color) {
		StringBuilder sb = new StringBuilder();
		sb.append("\\addplot[mark=x,draw=").append(color).append("] coordinates {\n");
		sb.append(store.getAsString(FAMILY_TO_PLOT));
		sb.append("};\n");
		sb.append("\\addlegendentry{").append(store.getAsString("dataset")).append("}\n");
		return sb.toString();
	}

	private static String combine(final List<String> one, final List<String> other) {
		StringBuilder sb = new StringBuilder();

		for (int i = 0; i < one.size(); i++) {
			if (!BINOMIAL && i >= GeneralConfig.POLYNOMIAL_SCALES.length) {
				break;
			}
			double oneVal = ValueUtil.round(Double.parseDouble(one.get(i)), 4);
			double otherVal = ValueUtil.round(Double.parseDouble(other.get(i)), 4);

			double x = (double) (i) / (one.size() - 1);
			x = GeneralConfig.POLYNOMIAL_SCALES[i];

			sb.append("(").append(x).append(",").append(ValueUtil.round(oneVal / otherVal, 4)).append(")\n");
		}

		return sb.toString();
	}

}
