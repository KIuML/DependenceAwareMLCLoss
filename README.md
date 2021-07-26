# Dependence-Aware Multi-Label Classification Loss

In this project the code for executing experiments for the dependence-aware multi-label classification loss is provided together with code to post-process the data for generating plots as presented in the corresponding paper.

Executing experiments requires a MySQL database as a backend for organization of the experiments.

# Setup and Running Experiments

1. Extract all the zip files containing the dataset splits in ``datasets/``
2. Fill out the ``experimenter.properties`` file to configure a database connection.
3. Open a command line and change into this directory.
4. Run ``.\gradlew setupExperimenter`` in order to add all the experiments configured in ``experimenter.properties``
5. In order to execute one of the experiments run ``.\gradlew runExperimenter``. Note that in order to run experiments a database connection to the previously created experiment table needs to be established.

# Post-Processing Result Data

Running experiments will produce files in ``out/`` containing prediction and ground-truth vectors per instance of the test data set. The respective vectors are stored in terms of ``.arff`` files as suggested by ``meka.core.Evaluation`` from MEKA. For further processing of the data, the arff files need to be parsed into a key-value store format of AILIbs which is done via executing the class ``de.lmu.dal.postprocessing.DataToKVStorePreparer``.

The already processed data (on which the results presented in the paper are based on is provided in the file) is located in the ``result-data/`` directory. This file can be taken as an input to the classes
* ``de.lmu.dal.postprocessing.DatasetWiseLossCurves``,
* ``de.lmu.dal.postprocessing.PairWiseLossCurves``,
* and ``de.lmu.dal.postprocessing.HeatMapData``

# Citing

Eyke Hüllermeier, Marcel Wever, Eneldo Loza Mencía, Johannes Fürnkranz, Michael Rapp: *A Flexible Class of Dependence-aware Multi-Label Loss Functions*
