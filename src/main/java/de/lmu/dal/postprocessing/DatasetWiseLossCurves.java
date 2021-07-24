package de.lmu.dal.postprocessing;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionOneLayerPartition;
import ai.libs.jaicore.basic.kvstore.KVStoreSequentialComparator;

public class DatasetWiseLossCurves {

	private static final List<String> ALGOS = Arrays.asList("cc", "clus-ensemble", "clus", "br", "rakel-2", "rakel-3", "rakel-4", "rakel-5", "lc", "boomer_label_wise_logistic_loss", "boomer_example_wise_logistic_loss",
			"boomer_binomial_loss_2", "boomer_binomial_loss_3", "boomer_binomial_loss_4", "boomer_binomial_loss_5", "boomer_binomial_loss_6", "boomer_binomial_loss_7", "boomer_binomial_loss_8", "boomer_binomial_loss_9",
			"boomer_binomial_loss_10", "boomer_binomial_loss_11", "boomer_binomial_loss_12", "boomer_binomial_loss_13", "boomer_binomial_2", "boomer_binomial_3", "boomer_binomial_4", "boomer_binomial_5", "boomer_binomial_6",
			"boomer_binomial_7", "boomer_binomial_8", "boomer_binomial_9", "boomer_binomial_10", "boomer_binomial_11", "boomer_binomial_12", "boomer_binomial_13");
	private static final List<String> DATASETS = Arrays.asList("birds", "emotions", "enron-f", "flags", "genbase", "llog-f", "medical", "scene", "yeast");

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
		replaceNames.put("boomer_label_wise_logistic_loss", "Boomer-1");
		replaceNames.put("boomer_example_wise_logistic_loss", "Boomer-K");
		replaceNames.put("boomer_binomial_loss_2", "Boomer-2");
		replaceNames.put("boomer_binomial_loss_3", "Boomer-3");
		replaceNames.put("boomer_binomial_loss_4", "Boomer-4");
		replaceNames.put("boomer_binomial_loss_5", "Boomer-5");
		replaceNames.put("boomer_binomial_loss_6", "Boomer-6");
		replaceNames.put("boomer_binomial_loss_7", "Boomer-7");
		replaceNames.put("boomer_binomial_loss_8", "Boomer-8");
		replaceNames.put("boomer_binomial_loss_9", "Boomer-9");
		replaceNames.put("boomer_binomial_loss_10", "Boomer-10");
		replaceNames.put("boomer_binomial_loss_11", "Boomer-11");
		replaceNames.put("boomer_binomial_loss_12", "Boomer-12");
		replaceNames.put("boomer_binomial_loss_13", "Boomer-13");
		replaceNames.put("boomer_binomial_2", "Boomer-2");
		replaceNames.put("boomer_binomial_3", "Boomer-3");
		replaceNames.put("boomer_binomial_4", "Boomer-4");
		replaceNames.put("boomer_binomial_5", "Boomer-5");
		replaceNames.put("boomer_binomial_6", "Boomer-6");
		replaceNames.put("boomer_binomial_7", "Boomer-7");
		replaceNames.put("boomer_binomial_8", "Boomer-8");
		replaceNames.put("boomer_binomial_9", "Boomer-9");
		replaceNames.put("boomer_binomial_10", "Boomer-10");
		replaceNames.put("boomer_binomial_11", "Boomer-11");
		replaceNames.put("boomer_binomial_12", "Boomer-12");
		replaceNames.put("boomer_binomial_13", "Boomer-13");

		KVStoreCollection col = new KVStoreCollection(FileUtils.readFileToString(new File("results.kvstore")));
		col.addAll(new KVStoreCollection(FileUtils.readFileToString(new File("results_boomer_v3.kvstore"))));

		KVStoreCollection filtered = new KVStoreCollection();
		col.stream().filter(x -> ALGOS.contains(x.getAsString("algorithm")) && DATASETS.contains(x.getAsString("dataset"))).forEach(filtered::add);

		Set<String> datasets = new HashSet<>();
		Set<String> algos = new HashSet<>();
		for (IKVStore s : filtered) {
			datasets.add(s.getAsString("dataset"));
			algos.add(replaceNames.get(s.getAsString("algorithm")));

			if (s.getAsString("algorithm").trim().equals("")) {
				System.out.println(s);
			}
		}

		filtered.stream().forEach(x -> x.put("algorithm", replaceNames.get(x.getAsString("algorithm"))));

		KVStoreCollectionOneLayerPartition comparisonPartition = new KVStoreCollectionOneLayerPartition("dataset", filtered);
		StringBuilder sb = new StringBuilder();

		int counter = 1;
		for (Entry<String, KVStoreCollection> part : comparisonPartition) {
			if (part.getValue().size() < 2) {
				continue;
			}
			System.out.println(part.getKey() + " " + (counter++));

			col = part.getValue();
			col.sort(new KVStoreSequentialComparator("algorithm"));

			// output the plots as a tickz picture
			sb.append("% ").append(part.getKey()).append("\n");
			sb.append("\\begin{my}\n");
			sb.append("\\begin{tikzpicture}\n").append("\\begin{axis}[xmin=1");
			if (BINOMIAL) {
				sb.append(",xmax=").append(col.get(0).getAsString("L"));
			} else {
				sb.append(",xmax=1000,xmode=log");
			}
			sb.append(",ymin=0,legend pos=outer north east,extra y ticks={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}, tick style={grid=major},xlabel=");
			if (BINOMIAL) {
				sb.append("$k$");
			} else {
				sb.append("$\\alpha$");
			}
			sb.append(",ylabel=loss (");
			if (BINOMIAL) {
				sb.append("$\\ell_{bin}$");
			} else {
				sb.append("$\\ell_{pol}$");
			}
			sb.append(")]\n");
			for (int i = 0; i < col.size(); i++) {
				sb.append(storeToPlot(col.get(i), "color" + (i + 1)));
			}
			sb.append("\\legend{};\n");
			sb.append("\\end{axis}\n").append("\\end{tikzpicture}\n").append("\\end{my}\n\n");
		}

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File("comparison-plots/datasetwise-" + FAMILY_TO_PLOT + "-notrim.tex")))) {
			bw.write(frameDoc(sb.toString()));
			// bw.write(sb.toString());
		}

	}

	private static String frameDoc(final String docString) {
		StringBuilder sb = new StringBuilder();

		sb.append("\\documentclass[multi=my,crop]{standalone}\n");

		sb.append("\\usepackage[utf8]{inputenc}\n");
		Arrays.asList("graphicx", "tikz", "pgfplots").stream().map(x -> "\\usepackage{" + x + "}\n").forEach(sb::append);

		sb.append("\\definecolor{color1}{HTML}{023EFF}\n" + "\\definecolor{color2}{HTML}{CCCCCC}\n" + "\\definecolor{color3}{HTML}{000000}\n" + "\\definecolor{color10}{HTML}{FF7C00}\n" + "\\definecolor{color11}{HTML}{1AC938}\n"
				+ "\\definecolor{color4}{HTML}{E8000B}\n" + "\\definecolor{color5}{HTML}{8B2BE2}\n" + "\\definecolor{color6}{HTML}{9F4800}\n" + "\\definecolor{color7}{HTML}{F14CC1}\n" + "\\definecolor{color8}{HTML}{A3A3A3}\n"
				+ "\\definecolor{color9}{HTML}{FFC400}\n" + "\\definecolor{color9}{HTML}{00D7FF}\n");

		sb.append("\\pgfplotsset{every axis plot/.append style={densely dashed,thick}}\n");

		sb.append("\n\\begin{document}\n\n");
		sb.append(docString).append("\n");
		sb.append("\n\\end{document}\n");

		return sb.toString();
	}

	private static String storeToPlot(final IKVStore store, final String color) {
		StringBuilder sb = new StringBuilder();
		sb.append("\\addplot[mark=none,draw=").append(color).append("] coordinates {\n");

		List<String> list = store.getAsStringList(FAMILY_TO_PLOT);
		for (int i = 0; i < list.size(); i++) {
			if (BINOMIAL) {
				sb.append("(").append(i + 1).append(",").append(list.get(i)).append(")\n");
			} else {
				sb.append("(").append(GeneralConfig.POLYNOMIAL_SCALES[i]).append(",").append(list.get(i)).append(")\n");
			}
		}
		sb.append("};\n");
		sb.append("\\addlegendentry{").append(store.getAsString("algorithm")).append("}\n");
		return sb.toString();
	}
}
