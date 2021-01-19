import os
import pandas as pd
from mcod_prep.launch_burden_calculator import BurdenCalculator
from mcod_prep.utils.causes import get_most_detailed_inj_causes
from cod_prep.claude.configurator import Configurator

CONF = Configurator('standard')


def check_output_exists(intermediate_causes, description, processes,
                        cache_kwargs=None, year_id=None):
    """Check that all expected outputs exist."""

    # use launch year ids if none are passed
    if year_id is None:
        year_id = BurdenCalculator.default_year_list

    # convert arguments to a list if they are not already
    args_dict = locals()
    for key, value in list(args_dict.items()):
        if (type(value) != list) and (key != 'description'):
            args_dict.update({key: [value]})

    # set each argument now that they've been converted to the correct type
    intermediate_causes = args_dict['intermediate_causes']
    processes = args_dict['processes']
    year_id = args_dict['year_id']

    # validate arguments to match the launch
    assert len(
        set(intermediate_causes) - set(BurdenCalculator.valid_intermediate_causes)
    ) == 0, "You passed an invalid intermediate cause"
    assert len(
        set(processes) - set(BurdenCalculator.process_order)
    ) == 0, "You passed an invalid process"
    process_suffix_dict = {
        'predict_fractions': '', 'calculate_deaths': '_deaths',
        'age_loc_aggregation': '_aggregate', 'cause_aggregation': '_aggregate'
    }

    # loop through processes, intermediate causes, years, to find any missing files
    missing_processes = {}
    for process in processes:
        suffix = process_suffix_dict[process]
        missing_int_causes = {}
        for int_cause in intermediate_causes:
            parent_dir = os.path.join(CONF.get_directory('process_data'), int_cause, description)
            # target list is unique to intermediate cause + model run

            if int_cause in ['x59', 'y34']:
                cause_list = get_most_detailed_inj_causes(
                    cache_kwargs, int_cause,
                    cause_set_version_id=CONF.get_id('reporting_cause_set_version'))
            else:
                cause_list = list(
                    pd.read_csv(os.path.join(parent_dir, "cause_list.csv")).keep_causes.unique()
                )
            if process == 'cause_aggregation':
                cause_list = ['_all_int_cause']
            year_cause_dict = {}
            for year in year_id:
                missing_causes = []
                for cause in cause_list:
                    filepath = parent_dir + '/{}/{}{}.csv'.format(year, cause, suffix)
                    if os.path.exists(filepath):
                        size = os.path.getsize(filepath)
                        # date = time.ctime(os.path.getmtime(filepath))
                        if size <= 0:
                            missing_causes.append(cause)
                            year_cause_dict.update({year: missing_causes})
                            missing_int_causes.update({int_cause: year_cause_dict})
                            missing_processes.update({process: missing_int_causes})
                    else:
                        missing_causes.append(cause)
                        year_cause_dict.update({year: missing_causes})
                        missing_int_causes.update({int_cause: year_cause_dict})
                        missing_processes.update({process: missing_int_causes})

    if len(missing_processes) > 0:
        print("The following year/causes were not available: {}".format(missing_processes))

    return missing_processes
