package ecmlpkdd2020;

import java.util.HashMap;
import java.util.Map;

import ai.libs.jaicore.experiments.Experiment;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import de.lmu.dal.experimenter.DependenceAwareMLCLossExperimenter;

public class ExperimenterTest {

	public static void main(final String[] args) throws ExperimentEvaluationFailedException, InterruptedException {

		DependenceAwareMLCLossExperimenter experimenter = new DependenceAwareMLCLossExperimenter();

		Map<String, String> valuesOfKeyFields = new HashMap<>();
		valuesOfKeyFields.put("dataset", "birds");
		valuesOfKeyFields.put("seed", "42");
		valuesOfKeyFields.put("split", "1");
		valuesOfKeyFields.put("algorithm", "clus");

		Experiment experiment = new Experiment(4096, 1, valuesOfKeyFields);

		ExperimentDBEntry experimentEntry = new ExperimentDBEntry(-1, experiment);

		experimenter.evaluate(experimentEntry, new IExperimentIntermediateResultProcessor() {
			@Override
			public void processResults(final Map<String, Object> results) {
				System.out.println("Results: " + results);
			}
		});
	}

}
