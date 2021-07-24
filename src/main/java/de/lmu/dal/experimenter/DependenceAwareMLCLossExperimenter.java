package de.lmu.dal.experimenter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;

import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import clus.Clus;
import meka.classifiers.MultiXClassifier;
import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.FW;
import meka.classifiers.multilabel.LC;
import meka.classifiers.multilabel.RAkEL;
import meka.core.F;
import meka.core.MLUtils;
import meka.core.Result;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.clus.ClusWrapperClassification;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelNodeImpl;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RenameAttribute;

public class DependenceAwareMLCLossExperimenter implements IExperimentSetEvaluator {

	/**
	 * Variables for the experiment and database setup
	 */
	private static final File configFile = new File("experimenter.properties");
	private static final IDependenceAwareMLCLossExperimenterConfig m = (IDependenceAwareMLCLossExperimenterConfig) ConfigCache.getOrCreate(IDependenceAwareMLCLossExperimenterConfig.class).loadPropertiesFromFile(configFile);
	private static final IDatabaseConfig dbconfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(configFile);
	private static final IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(dbconfig);

	public static void main(final String[] args)
			throws ExperimentDBInteractionFailedException, AlgorithmTimeoutedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		if (args.length > 0) {
			switch (args[0]) {
			case "init":
				createTableWithExperiments();
				break;
			case "run":
				runExperiments();
				break;
			case "delete":
				deleteTable();
				break;
			}
		} else {
			runExperiments();
		}
	}

	public static void createTableWithExperiments()
			throws ExperimentDBInteractionFailedException, AlgorithmTimeoutedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(m, dbHandle);
		preparer.synchronizeExperiments();
	}

	public static void deleteTable() throws ExperimentDBInteractionFailedException {
		dbHandle.deleteDatabase();
	}

	public static void runExperiments() throws ExperimentDBInteractionFailedException, InterruptedException {
		ExperimentRunner runner = new ExperimentRunner(m, new DependenceAwareMLCLossExperimenter(), dbHandle);
		runner.randomlyConductExperiments(-1);
	}

	private String datasetName;
	private Long seed;
	private String split;
	private Instances trainDataset = null;
	private Instances testDataset = null;

	private void clear() {
		this.trainDataset = null;
		this.testDataset = null;
		this.datasetName = null;
		this.seed = null;
		this.split = null;
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException {
		/* get experiment setup */
		this.clear();
		Map<String, String> description = experimentEntry.getExperiment().getValuesOfKeyFields();
		String algorithm = description.get("algorithm");
		this.datasetName = description.get("dataset");
		this.seed = Long.parseLong(description.get("seed"));
		this.split = description.get("split");

		String algoName = "";
		switch (algorithm) {
		case "rakel":
			algoName = RAkEL.class.getName();
			break;
		case "br":
			algoName = BR.class.getName();
			break;
		case "lc":
			algoName = LC.class.getName();
			break;
		case "clus":
			algoName = ClusWrapperClassification.class.getName();
			break;
		}
		System.out.println("Experiment #" + experimentEntry.getId() + ": Evaluate " + algoName + " on " + this.datasetName + " split " + this.split + " and seed " + this.seed);

		Map<String, Integer> numLabels = new HashMap<>();
		numLabels.put("arts1", 26);
		numLabels.put("bibtex", 159);
		numLabels.put("birds", 19);
		numLabels.put("bookmarks", 208);
		numLabels.put("business1", 30);
		numLabels.put("computers1", 33);
		numLabels.put("education1", 33);
		numLabels.put("emotions", 6);
		numLabels.put("enron-f", 53);
		numLabels.put("entertainment1", 21);
		numLabels.put("flags", 12);
		numLabels.put("genbase", 27);
		numLabels.put("health1", 32);
		numLabels.put("llog-f", 75);
		numLabels.put("mediamill", 101);
		numLabels.put("medical", 45);
		numLabels.put("recreation1", 22);
		numLabels.put("reference1", 33);
		numLabels.put("scene", 6);
		numLabels.put("science1", 40);
		numLabels.put("social1", 39);
		numLabels.put("society1", 27);
		numLabels.put("tmc2007", 22);
		numLabels.put("yeast", 14);

		try {
			StringBuilder outputFileNamePrefix = new StringBuilder();
			outputFileNamePrefix.append(this.datasetName).append(File.separator).append(SetUtil.implode(Arrays.asList(this.datasetName, this.split, this.seed, algorithm), "-"));

			int L = numLabels.get(this.datasetName);

			Map<String, Object> output = new HashMap<>();

			System.out.println("Create algorithm for " + algorithm + "and evaluate it");
			switch (algorithm) {
			case "br": {
				System.out.println("Create BR classifier");
				BR br = new BR();
				br.setClassifier(new J48());
				StringBuilder outputFileName = new StringBuilder(outputFileNamePrefix);
				outputFileName.append(".arff");
				System.out.println("Evaluate BR classifier");
				this.evaluateModel(br, new File(m.getOutputFolder(), outputFileName.toString()));
				break;
			}
			case "lc": {
				System.out.println("Create LC classifier");
				LC lc = new LC();
				lc.setClassifier(new J48());
				StringBuilder outputFileName = new StringBuilder(outputFileNamePrefix);
				outputFileName.append(".arff");
				System.out.println("Evaluate LC classifier");
				this.evaluateModel(lc, new File(m.getOutputFolder(), outputFileName.toString()));
				break;
			}
			case "rakel": {
				Set<Integer> ks = new HashSet<>();
				IntStream.range(1, 6).forEach(ks::add);
				ks.add((int) Math.floor((double) L / 2));
				ks.add(L - 1);
				for (int k : ks) {
					System.gc();
					System.out.println("Create RAkEL classifier with k=" + k);
					RAkEL rakel = new RAkEL();
					rakel.setK(k);
					rakel.setM(L - k + 1);
					rakel.setClassifier(new J48());

					StringBuilder outputFileName = new StringBuilder(outputFileNamePrefix);
					outputFileName.append("-").append(k).append(".arff");

					System.out.println("Evaluate RAkEL classifier with k=" + k);
					this.evaluateModel(rakel, new File(m.getOutputFolder(), outputFileName.toString()));
					output.put("progress", k + "/" + L + " (" + ValueUtil.round(((double) k / L) * 100, 2) + "%)");
					processor.processResults(output);
				}
			}
			case "rakel2": {
				Set<Integer> ks = new HashSet<>();
				IntStream.range(2, 6).forEach(ks::add);
				// ks.add((int) Math.floor((double) L / 2));
				// ks.add(L - 1);
				for (int k : ks) {
					System.gc();
					System.out.println("Create RAkEL classifier with k=" + k);
					RAkEL rakel = new RAkEL();
					rakel.setK(k);
					rakel.setM((int) ((3.0 * L) / k));
					rakel.setClassifier(new J48());

					StringBuilder outputFileName = new StringBuilder(outputFileNamePrefix);
					outputFileName.append("-").append(k).append(".arff");

					System.out.println("Evaluate RAkEL classifier with k=" + k);
					this.evaluateModel(rakel, new File(m.getOutputFolder(), outputFileName.toString()));
					output.put("progress", k + "/" + L + " (" + ValueUtil.round(((double) k / L) * 100, 2) + "%)");
					processor.processResults(output);
				}
			}
			case "cc": {
				System.out.println("Create CC classifier");
				CC cc = new CC();
				StringBuilder outputFileName = new StringBuilder(outputFileNamePrefix);
				outputFileName.append(".arff");
				cc.setClassifier(new J48());
				System.out.println("Evaluate CC classifier");
				this.evaluateModel(cc, new File(m.getOutputFolder(), outputFileName.toString()));
			}
			case "fw": {
				System.out.println("Create FW classifier");
				FW fw = new FW();
				StringBuilder outputFileName = new StringBuilder(outputFileNamePrefix);
				outputFileName.append(".arff");
				fw.setClassifier(new J48());
				System.out.println("Evaluate FW classifier");
				this.evaluateModel(fw, new File(m.getOutputFolder(), outputFileName.toString()));
			}
			case "clus": {
				System.out.println("Create Clus classifier");
				boolean[] ensemble = { true, false };

				for (boolean isEnsemble : ensemble) {
					File workingDir = new File("tmp/" + this.datasetName + "-" + this.seed + "-" + this.split + "-" + isEnsemble + "/");
					workingDir.mkdirs();
					try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File(workingDir, "config.s")))) {
						bw.write("[Data]\nFile\nTestSet\n[Attributes]\nTarget\n[Output]\nWritePredictions=Test");
					}

					ClusWrapperClassification clus = new ClusWrapperClassification(workingDir.getPath() + "/", this.datasetName, workingDir.getPath() + "/config.s");
					clus.setEnsemble(isEnsemble);

					StringBuilder outputFileName = new StringBuilder(outputFileNamePrefix);
					if (isEnsemble) {
						outputFileName.append("-ensemble");
					}
					outputFileName.append(".arff");

					System.out.println("Evaluate Clus classifier");
					this.evaluateModel(clus, new File(m.getOutputFolder(), outputFileName.toString()), isEnsemble ? 10.0 : 1.0);
				}
			}

			}

			output.put("progress", "done");
			processor.processResults(output);
		} catch (Throwable e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	private Instances getDataset(final boolean train) throws ExperimentEvaluationFailedException {
		if (train) {
			try {
				if (this.trainDataset == null) {
					this.trainDataset = new Instances(new FileReader(new File(m.getDatasetFolder(), SetUtil.implode(Arrays.asList(this.datasetName, this.seed + "", this.split, "train"), "_") + ".arff")));

					String relationName = this.trainDataset.relationName();
					RenameAttribute ra = new RenameAttribute();
					ra.setAttributeIndices("first-last");
					ra.setFind(" ");
					ra.setReplaceAll(true);
					ra.setReplace("_");
					ra.setInputFormat(this.trainDataset);
					this.trainDataset = Filter.useFilter(this.trainDataset, ra);
					ra = new RenameAttribute();
					ra.setAttributeIndices("first-last");
					ra.setFind("'");
					ra.setReplaceAll(true);
					ra.setReplace("");
					ra.setInputFormat(this.trainDataset);
					this.trainDataset = Filter.useFilter(this.trainDataset, ra);

					this.trainDataset.setRelationName(relationName);
					MLUtils.prepareData(this.trainDataset);
				}
				return new Instances(this.trainDataset);
			} catch (Exception e) {
				throw new ExperimentEvaluationFailedException("Could not load train dataset", e);
			}
		} else {
			try {
				if (this.testDataset == null) {
					this.testDataset = new Instances(new FileReader(new File(m.getDatasetFolder(), SetUtil.implode(Arrays.asList(this.datasetName, this.seed + "", this.split, "test"), "_") + ".arff")));

					String relationName = this.testDataset.relationName();
					RenameAttribute ra = new RenameAttribute();
					ra.setAttributeIndices("first-last");
					ra.setFind(" ");
					ra.setReplaceAll(true);
					ra.setReplace("_");
					ra.setInputFormat(this.testDataset);
					this.testDataset = Filter.useFilter(this.testDataset, ra);
					ra = new RenameAttribute();
					ra.setAttributeIndices("first-last");
					ra.setFind("'");
					ra.setReplaceAll(true);
					ra.setReplace("");
					ra.setInputFormat(this.testDataset);
					this.testDataset = Filter.useFilter(this.testDataset, ra);

					this.testDataset.setRelationName(relationName);
					MLUtils.prepareData(this.testDataset);
				}
				return new Instances(this.testDataset);
			} catch (Exception e) {
				throw new ExperimentEvaluationFailedException("Could not load test dataset", e);
			}

		}
	}

	private MultiLabelInstances meka2mulan(final Instances mekaInstances) throws InvalidDataFormatException {
		int classIndex = mekaInstances.classIndex();
		int numAttributes = mekaInstances.numAttributes();
		Instances convertedToMulan = F.meka2mulan(new Instances(mekaInstances), mekaInstances.classIndex());
		LabelsMetaDataImpl labelsMetaData = new LabelsMetaDataImpl();
		IntStream.range(numAttributes - classIndex, numAttributes).mapToObj(x -> new LabelNodeImpl(convertedToMulan.attribute(x).name())).forEach(labelsMetaData::addRootNode);
		MultiLabelInstances mulanInstances = new MultiLabelInstances(convertedToMulan, labelsMetaData);
		return mulanInstances;
	}

	private void evaluateModel(final ClusWrapperClassification learner, final File outputFile, final double normalizeBy) throws ExperimentEvaluationFailedException {
		try {
			System.out.println("Build mulan MultiLabelLearner " + learner.getClass().getSimpleName());
			learner.build(this.meka2mulan(this.getDataset(true)));
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}

		try {
			System.out.println("Make predictions with mulan MultiLabelLearner " + learner.getClass().getSimpleName());
			// MEKA Result
			Instances mekaTestData = this.getDataset(false);
			Result res = new Result(mekaTestData.size(), mekaTestData.classIndex());

			// Mulan evaluation for CLUS
			MultiLabelInstances testData = this.meka2mulan(mekaTestData);
			boolean isEnsemble = learner.isEnsemble();
			boolean isRuleBased = learner.isRuleBased();
			boolean isRegression;
			MultiLabelOutput output = learner.makePrediction(testData.getDataSet().instance(0));
			if (output.hasPvalues()) {
				isRegression = true;
			} else {
				isRegression = false;
			}

			String clusWorkingDir = learner.getClusWorkingDir();
			String datasetName = learner.getDatasetName();
			// write the supplied MultilabelInstances object in an arff formated file (accepted by CLUS)
			ClusWrapperClassification.makeClusCompliant(testData, clusWorkingDir + datasetName + "-test.arff");

			// call Clus.main to write the output files!
			ArrayList<String> clusArgsList = new ArrayList<String>();
			if (isEnsemble) {
				clusArgsList.add("-forest");
			}
			if (isRuleBased) {
				clusArgsList.add("-rules");
			}
			// the next argument passed to Clus is the settings file!
			clusArgsList.add(clusWorkingDir + datasetName + "-train.s");
			System.out.println(clusArgsList);
			String[] clusArgs = clusArgsList.toArray(new String[clusArgsList.size()]);
			Clus.main(clusArgs);

			System.out.println("===");

			// then parse the output files and finally update the measures!
			// open and load the test set predictions file, which is in arff format
			String predictionsFilePath = clusWorkingDir + datasetName + "-train.test.pred.arff";
			BufferedReader reader = new BufferedReader(new FileReader(predictionsFilePath));
			Instances predictionInstances = new Instances(reader);
			reader.close();

			Instances testDataset = testData.getDataSet();

			int numInstances = testDataset.numInstances();
			for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
				Instance instance = testDataset.instance(instanceIndex);
				if (testData.hasMissingLabels(instance)) {
					continue;
				}
				Instance labelsMissing = (Instance) instance.copy();
				labelsMissing.setDataset(instance.dataset());
				for (int i = 0; i < testData.getNumLabels(); i++) {
					labelsMissing.setMissing(testData.getLabelIndices()[i]);
				}
				// clus way
				Instance predictionInstance = predictionInstances.instance(instanceIndex);
				double[] predictionsPerSample = new double[testData.getNumLabels()];
				int k = 0;
				for (int j = 0; j < predictionInstance.numValues() - 1; j++) {
					// collect predicted values
					if (isRegression) {
						if (isEnsemble && !isRuleBased) {
							if (j >= (testData.getNumLabels())) {
								predictionsPerSample[k] = predictionInstance.value(j);
								k++;
							}
						} else {
							if (j >= (testData.getNumLabels() * 2 + 1)) {
								predictionsPerSample[k] = predictionInstance.value(j);
								k++;
							}
						}
					} else {
						if (isEnsemble && !isRuleBased) {
							if (j >= testData.getNumLabels() * 2) {
								predictionsPerSample[k] = predictionInstance.value(j);
								j++;
								k++;
							}
						} else {
							if (j >= (testData.getNumLabels() * 5 + 1)) {
								predictionsPerSample[k] = predictionInstance.value(j) / (predictionInstance.value(j) + predictionInstance.value(j + 1));
								j++;
								k++;
							}
						}
					}
					if (k == testData.getNumLabels()) {
						break;
					}
				}

				IntStream.range(0, predictionsPerSample.length).forEach(x -> predictionsPerSample[x] /= normalizeBy);
				res.addResult(predictionsPerSample, mekaTestData.get(instanceIndex));
			}

			System.out.println(Result.getPredictionsAsInstances(res));
			System.out.println("Write results to file...");
			try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile))) {
				bw.write(Result.getPredictionsAsInstances(res).toString());
			}
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}

	}

	private void evaluateModel(final MultiXClassifier c, final File outputFile) throws ExperimentEvaluationFailedException {
		if (outputFile.exists()) {
			System.out.println(outputFile.getAbsolutePath() + " already exists. Skipping this model evaluation.");
			return;
		}

		try {
			// ensure parent dir is available
			outputFile.getParentFile().mkdirs();
			System.out.println("Evaluate classifier on train test split...");
			Result res = Evaluation.evaluateModel(c, this.getDataset(true), this.getDataset(false));
			System.out.println("Write results to file...");
			try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile))) {
				bw.write(Result.getPredictionsAsInstances(res).toString());
			}
		} catch (ExperimentEvaluationFailedException e) {
			throw e;
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}
}
