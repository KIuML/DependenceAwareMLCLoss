package de.lmu.dal.experimenter;

import java.io.File;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.IExperimentSetConfig;

@Sources({ "file:experimenter.properties" })
public interface IDependenceAwareMLCLossExperimenterConfig extends IExperimentSetConfig, IDatabaseConfig {

	public static final String K_DATASET_FOLDER = "datasetFolder";
	public static final String K_OUTPUT_FOLDER = "outputFolder";

	@Key(K_DATASET_FOLDER)
	public File getDatasetFolder();

	@Key(K_OUTPUT_FOLDER)
	public File getOutputFolder();

}
