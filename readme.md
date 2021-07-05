# Pipeline for processing multiple cause data #

Check out our [paper](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01501-1) for more information on how multiple cause of death data are being used to improve data quality and make important advancements in cause-specific mortality estimates.

Code is designed to mirror claude for ease of transitioning
between data processing pipelines on CoD and MCause<sup>1</sup>.

There are two main sets of processes, divided into two launch scripts:

1) [This script](launch_mcause_data_prep.py) launches phases related to formatting data and pre-modeling steps (i.e. formatting, mapping to ICD codes, redistributing garbage deaths)

2) [This script](launch_burden_calculator.py) launches modeling + post-modeling pieces (i.e. run_model, predictions, location/age/cause aggregation, creation of redistribution proportions, mortality, incidence)

These two launch scripts were separated out for ease of flexibility across projects.
Not all projects have modeling pieces, so sometimes it is helpful
to only look at formatting & mapping related to a specific intermediate cause of interest.

<sup>1</sup>this code began with only multiple cause of death VR data and has been
expanded to include hospital data containing multiple diagnoses and
hospital data linked with VR data. For this reason, you may find discrepancies
in naming between "mcod" and "mcause", however, they are conceptually interchangeable.
