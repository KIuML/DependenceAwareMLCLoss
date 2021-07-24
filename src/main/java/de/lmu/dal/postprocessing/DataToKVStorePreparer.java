package de.lmu.dal.postprocessing;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.api4.java.ai.ml.classification.multilabel.evaluation.IMultiLabelClassification;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.ArrayUtil;
import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.ml.classification.multilabel.MultiLabelClassification;
import ai.libs.jaicore.ml.classification.multilabel.evaluation.loss.nonadditive.OWARelevanceLoss;
import ai.libs.jaicore.ml.classification.multilabel.evaluation.loss.nonadditive.owa.MoebiusTransformOWAValueFunction;
import ai.libs.jaicore.ml.classification.multilabel.evaluation.loss.nonadditive.owa.PolynomialOWAValueFunction;
import weka.core.Instances;

public class DataToKVStorePreparer {

	private static final double THRESHOLD = 0.5;
	private static final String SIZE_LABEL = "size";

	private static final File INPUT_DIR = new File("out/");
	private static final File OUTPUT_FILE = new File("result-data/results.kvstore");

	class AlgoEntry {
		private IKVStore store;
		private List<File> files = new ArrayList<>(10);

		public AlgoEntry(final IKVStore store) {
			this.store = store;
		}

		public Instances getData() throws IOException {
			Instances data = new Instances(new FileReader(this.files.get(0)));
			for (int i = 1; i < this.files.size(); i++) {
				data.addAll(new Instances(new FileReader(this.files.get(i))));
			}
			return data;
		}

		public void addFile(final File file) {
			this.files.add(file);
			this.store.put(SIZE_LABEL, this.files.size() + "");
		}

		public IKVStore getStore() {
			return this.store;
		}

		public int size() {
			return this.files.size();
		}
	}

	public DataToKVStorePreparer() throws IOException {
		KVStoreCollection col = new KVStoreCollection("collectinID=mlcloss_results");
		if (OUTPUT_FILE.exists()) {
			col.addAll(new KVStoreCollection(FileUtils.readFileToString(OUTPUT_FILE)));
		}

		for (File datasetFolder : INPUT_DIR.listFiles()) {
			if (datasetFolder.getName().equals("backup")) {
				continue;
			}

			Map<String, AlgoEntry> algoMap = new HashMap<>();
			for (File resultFile : datasetFolder.listFiles()) {
				String[] descriptor = resultFile.getName().substring(0, resultFile.getName().length() - 5).split("-");
				IKVStore store = new KVStore();

				int i;
				List<String> datasetNameList = new ArrayList<>();
				for (i = 0; i < StringUtils.countMatches(datasetFolder.getName(), "-") + 1; i++) {
					datasetNameList.add(descriptor[i]);
				}
				store.put("dataset", SetUtil.implode(datasetNameList, "-"));
				i++; // skip split id
				store.put("seed", descriptor[i++]);
				store.put("algorithm", IntStream.range(i, descriptor.length).mapToObj(x -> descriptor[x]).collect(Collectors.joining("-")));
				algoMap.computeIfAbsent(store.getAsString("algorithm"), t -> new AlgoEntry(store)).addFile(resultFile);
			}

			for (Entry<String, AlgoEntry> algoEntry : algoMap.entrySet()) {
				IKVStore store = algoEntry.getValue().getStore();

				Map<String, String> selection = new HashMap<>();
				Arrays.asList("dataset", "seed", "algorithm").stream().forEach(x -> selection.put(x, store.getAsString(x)));
				KVStoreCollection selectCol = col.select(selection);
				if (!selectCol.isEmpty() && selectCol.get(0).getAsInt("size") >= store.getAsInt("size")) {
					continue;
				}
				if (!selectCol.isEmpty()) {
					col.remove(selectCol.get(0));
					System.out.println("Update moebius and polynomial loss traces for " + algoEntry.getValue().getStore());
				} else {
					System.out.println("Compute moebius and polynomial loss traces for " + algoEntry.getValue().getStore());
				}

				Pair<int[][], int[][]> gtAndPred = extractGTandPrediction(algoEntry.getValue().getData());
				store.put("L", gtAndPred.getX()[0].length);

				String moebiusSeries = Arrays.stream(evaluateMoebiusTransformRelevanceLoss(gtAndPred.getX(), gtAndPred.getY())).mapToObj(x -> x + "").collect(Collectors.joining(","));
				String polySeries = Arrays.stream(evaluatePolynomialRelevanceLoss(gtAndPred.getX(), gtAndPred.getY())).mapToObj(x -> x + "").collect(Collectors.joining(","));
				store.put("moebius", moebiusSeries);
				store.put("polynomial", polySeries);
				col.add(store);
			}

			col.serializeTo(OUTPUT_FILE);
		}
	}

	private static Pair<int[][], int[][]> extractGTandPrediction(final Instances data) {
		int[][] gt = new int[data.size()][];
		double[][] pred = new double[data.size()][];

		int numAttributes = data.numAttributes();
		for (int i = 0; i < data.size(); i++) {
			int currentI = i;
			gt[i] = IntStream.range(0, numAttributes / 2).map(x -> (int) data.get(currentI).value(x)).toArray();
			pred[i] = IntStream.range(numAttributes / 2, numAttributes).mapToDouble(x -> data.get(currentI).value(x)).toArray();
		}
		return new Pair<>(gt, ArrayUtil.thresholdDoubleToBinaryMatrix(pred, THRESHOLD));
	}

	private static double[] evaluateMoebiusTransformRelevanceLoss(final int[][] gt, final int[][] pred) {
		double[] lossValues = new double[gt[0].length];
		for (int k = 1; k <= gt[0].length; k++) {
			OWARelevanceLoss l = new OWARelevanceLoss(new MoebiusTransformOWAValueFunction(k));
			lossValues[k - 1] = l.loss(Arrays.stream(gt).collect(Collectors.toList()), toClassifications(pred));
		}
		return lossValues;
	}

	private static double[] evaluatePolynomialRelevanceLoss(final int[][] gt, final int[][] pred) {
		double[] lossValues = new double[GeneralConfig.POLYNOMIAL_SCALES.length];
		for (int i = 0; i < GeneralConfig.POLYNOMIAL_SCALES.length; i++) {
			OWARelevanceLoss l = new OWARelevanceLoss(new PolynomialOWAValueFunction(GeneralConfig.POLYNOMIAL_SCALES[i]));
			lossValues[i] = l.loss(Arrays.stream(gt).collect(Collectors.toList()), toClassifications(pred));
		}
		return lossValues;
	}

	private static List<IMultiLabelClassification> toClassifications(final int[][] values) {
		List<IMultiLabelClassification> list = new ArrayList<>();
		for (int i = 0; i < values.length; i++) {
			list.add(new MultiLabelClassification(Arrays.stream(values[i]).mapToDouble(x -> x).toArray()));
		}
		return list;
	}

	public static void main(final String[] args) throws IOException {
		new DataToKVStorePreparer();
	}
}
