# Pipeline for processing all multiple cause data. #

Code is designed to mirror claude for ease of transitioning
between data processing pipelines on CoD and MCause. [^1]

There are two main sets of processes, divided into two launch scripts:

1) launch_mcause_data_prep.py: launches phases related to formatting data pre-modeling (e.g. run_phase_format_map, run_phase_redistribution)

2) launch_burden_calculator.py: launches modeling + post-modeling pieces (e.g. run_model, predictions, location/age/cause aggregation, creation of redistribution proportions, mortality, incidence)

These two launch scripts were separated out for ease of flexibility across projects.
Not all projects have modeling pieces, so sometimes it is helpful
to only look at formatting & mapping related to a specific intermediate cause of interest.

[^1] this code began with only multiple cause of death VR data and has been
expanded to include hospital data containing multiple diagnoses and
hospital data linked with VR data. For this reason, you may find discrepancies
in naming between "mcod" and "mcause", however, they are conceptually interchangeable.
# multiple_cause
