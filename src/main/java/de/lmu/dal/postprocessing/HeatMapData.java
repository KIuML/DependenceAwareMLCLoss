package de.lmu.dal.postprocessing;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionOneLayerPartition;

public class HeatMapData {

	public static void main(final String[] args) throws IOException {
		KVStoreCollection col = new KVStoreCollection(FileUtils.readFileToString(new File("result-data/results_boomer_v4.kvstore")));
		KVStoreCollectionOneLayerPartition partitions = new KVStoreCollectionOneLayerPartition("dataset", col);

		for (Entry<String, KVStoreCollection> part : partitions) {
			System.out.println(part.getKey());
			double[][] matrix = new double[part.getValue().size()][];
			for (IKVStore s : part.getValue()) {
				System.out.println(s);
				double[] moebius = s.getAsDoubleList("moebius").stream().mapToDouble(x -> x).toArray();
				if (s.getAsString("algorithm").equals("boomer_label_wise_logistic_loss")) {
					matrix[0] = moebius;
				} else if (s.getAsString("algorithm").equals("boomer_example_wise_logistic_loss")) {
					matrix[part.getValue().size() - 1] = moebius;
				} else {
					String[] split = s.getAsString("algorithm").split("\\_");
					matrix[Integer.parseInt(split[split.length - 1]) - 1] = moebius;
				}
			}

			double[][] diffMatrix = new double[matrix.length][matrix[0].length];
			for (int i = 0; i < diffMatrix.length; i++) {
				diffMatrix[i] = new double[matrix[0].length];
			}

			for (int j = 0; j < diffMatrix[0].length; j++) {
				Double min = matrix[0][j];
				for (int i = 1; i < diffMatrix.length; i++) {
					if (matrix[i][j] < min) {
						min = matrix[i][j];
					}
				}
				for (int i = 0; i < diffMatrix.length; i++) {
					diffMatrix[i][j] = matrix[i][j] - min;
				}
			}

			Arrays.stream(diffMatrix).map(Arrays::toString).forEach(System.out::println);
		}

	}

}
