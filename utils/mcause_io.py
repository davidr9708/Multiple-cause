import pandas as pd
import numpy as np
import warnings
import os
import re
from datetime import datetime, timedelta
from mcod_prep.utils.nids import get_datasets
from cod_prep.utils.misc import print_log_message
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import check_output_exists, get_phase_output, construct_phase_path

CONF = Configurator('standard')


def get_datestamp_modtime(filepath):
    return datetime.fromtimestamp(os.path.getmtime(filepath))


def get_launch_set_id_of_active(phase, nid, extract_type_id, sub_dirs=None):
    """Get the launch set id out of the archive for the 'current' file.

    Note: Haven't created the concept of launch_set_id for mcause work.
    We use the Y_M_D timestamp as a proxy.
    """
    phase_path = construct_phase_path(phase, nid, extract_type_id, sub_dirs=sub_dirs,
                                      file_format='csv')
    if not os.path.exists(phase_path):
        print(f"{phase_path} does not exist")
        return np.NaN
    curr_datestamp = get_datestamp_modtime(phase_path)

    data_folder = os.path.dirname(phase_path)
    arch_folder = os.path.join(data_folder, "_archive")
    arch_files = os.listdir(arch_folder)
    phase_arch_files = [f for f in arch_files if phase in f]

    close_time_launch_sets = []

    # shouldn't have write times longer than an hour
    time_elapsed = timedelta(0, 3600)

    for phase_arch_file in phase_arch_files:
        test_datestamp = get_datestamp_modtime(
            os.path.join(arch_folder, phase_arch_file)
        )

        if abs(curr_datestamp - test_datestamp) < time_elapsed:
            pattern = "{phase}_([0-9_]+)\.csv".format(phase=phase)
            lsid_search = re.search(pattern, phase_arch_file)
            assert lsid_search is not None, \
                "{} did not match {}".format(phase_arch_file, pattern)
            launch_set_id = lsid_search.group(1)
            close_time_launch_sets.append(launch_set_id)

    assert len(close_time_launch_sets) > 0, "Cannot determine launch set id,"\
                                            " the archive and current file"\
                                            " timestamps are too far apart"
    launch_set_id = max(close_time_launch_sets)
    return launch_set_id


def get_mcause_data(phase, nid=None, extract_type_id=None, source=None,
                    nid_extract_records=None, location_id=None,
                    sample=False, sample_num=None, force_rerun=False,
                    data_type_id=None, year_id=None, code_system_id=None,
                    iso3=None, region_id=None, launch_set_id=None,
                    location_set_id=None, location_set_version_id=None,
                    is_active=None, exec_function=None, exec_function_args=[],
                    block_rerun=True, where_filter=None, sub_dirs=None,
                    assert_all_available=False, verbose=False,
                    attach_launch_set_id=False, refresh_id=None,
                    usecols=None):
    """Combine datasets for given phase, optionally applying a function.

    Each dataset filter is additive, so if you provide arguments that filter
    to zero datasets it will return nothing.
        Arguments:
        phase, str: pull this phase. only one phase at a time for this method
        ***** dataset filters *****
        nid, int or list of ints: pull only these nids
        source, str or list of strs: pull only these sources
        nid_extract_records, list of 2-element nid, extract_type_id tuples
            This allows you to get the information in get_datasets for an
            already specified set of nid-extracts. Most useful for getting
            data after you've already done custom work to narrow down which
            nid-extracts you want - after a get_completion_status, for example.
        location_id, int or list of ints: pull only data for these locations
        data_type_id, int or list of ints: pull only these data types
        year_id, int or list of ints: pull only these years
        code_system_id, int or list of ints: pull only these code systems
        iso3, str: pull only this country
        is_active, bool: only active nid-extracts (some are not uploaded
            in production runs, like US counties)
        exec_function, runnable: run this function on each file that is read
            exec_function is useful if you are combining a ton of datasets
            that need to be filtered to one cause, or collapsed, or both. This
            function should take a claude-dataframe and return one
            DO NOT REMOVE THESE COLUMNS IN YOUR FUNCTION:
                ['location_id', 'year_id']
        exec_function_args, list: call exec_function with these args
            (one item in the list per function argument - so if the function
             has one argument, and that is a list, then exec_function should
             be a list with one item, which is itself a list, like:
             exec_function_args = [[my_list]]
        sample, bool: Whether to pick a number (=sample_num) of random datasets
            from those matching dataset filters. Uses pd.DataFrame.sample on
            the returned nid/extract pairs. May return an unpredictable number
            of rows if the picked datasets are subnational or have multiple
            years contained. Useful for training models on CoD data.
        sample_num, int: The number of datasets to sample if sample=True.
        force_rerun, bool: Whether to force a query of the nids database for
            parsing which datasets to read
        block_rerun, bool: Whether to prevent query of the nids database
            force & block can't both be True
        where_filter, str: should look like:
            "cause_id == 491" for pulling only cause_id 491. Will only
            work on cause_id
        assert_all_available, bool: fail if any of the matched nid/extracts
            were not available
        attach_launch_set_id, bool: add the launch_set_id of the active dataset
            (takes extra time as it does this by measuring completion
             times of the file; an additional ~1-7 ms per nid-extract)

    Returns:
        df, in claude format
    """
    if attach_launch_set_id:
        assert launch_set_id is None, \
            "You want to pull data for a previous launch set and attach " \
            "the launch set id for the current data; this information "\
            "is incompatible."

    if sample:
        assert sample_num is not None, \
            "If sampling datasets, must pass number of datasets to " \
            "sample with sample_num argument. Got: {}".format(sample_num)

    if launch_set_id:
        assert refresh_id is None, "Cannot specify both launch_set_id and refresh_id"

    dataset_filters = {
        'nid': nid,
        'extract_type_id': extract_type_id,
        'nid_extract_records': nid_extract_records,
        'source': source,
        'location_id': location_id,
        'year_id': year_id,
        'data_type_id': data_type_id,
        'code_system_id': code_system_id,
        'iso3': iso3,
        'region_id': region_id,
        'location_set_id': location_set_id,
        'location_set_version_id': location_set_version_id,
        'is_active': is_active,
    }
    if verbose:
        print_log_message("Getting datasets to read")

    datasets = get_datasets(
        force_rerun=force_rerun,
        block_rerun=block_rerun,
        verbose=verbose,
        **dataset_filters
    )
    if verbose:
        print_log_message("Got {} datasets".format(len(datasets)))

    pairs = datasets[['nid', 'extract_type_id']].drop_duplicates()

    if sample:
        if sample_num > len(pairs):
            warnings.warn(
                "Sample num of ({}) exceeded number of datasets "
                "({})".format(sample_num, len(pairs))
            )
            sample_num = len(pairs)
        pairs = pairs.sample(n=sample_num)
        if verbose:
            print_log_message("Restricted to {} random datasets".format(sample_num))
    # to_records returns records in type np.record instead of tuple - more
    # predictable behavior if this returns a list of tuples
    nid_extract_pairs = [tuple(pair) for pair in list(pairs.to_records(index=False))]

    if verbose:
        print_log_message("Checking which datasets have available files")
    avail_nid_extracts = []
    for nid, extract_type_id in nid_extract_pairs:
        if check_output_exists(phase, nid, extract_type_id,
                               sub_dirs=sub_dirs, refresh_id=refresh_id):
            avail_nid_extracts.append((nid, extract_type_id))

    bad_nid_extracts = list(set(nid_extract_pairs) - set(avail_nid_extracts))
    info_string = "Found {n} files to read data for.".format(n=len(avail_nid_extracts))
    if len(bad_nid_extracts) > 0:
        info_string = info_string + f" These nids were not available: \n{bad_nid_extracts}"
    if launch_set_id:
        info_string += " Using launch set id {}".format(launch_set_id)
    if assert_all_available and len(bad_nid_extracts) != 0:
        raise AssertionError(info_string)
    elif verbose:
        print_log_message(info_string)

    if len(avail_nid_extracts) == 0:
        raise AssertionError(
            "No files were available with given dataset "
            "filters: \n{}".format(dataset_filters)
        )
    if verbose:
        print_log_message(
            "Reading and appending {} data for {} "
            "nid-extracts".format(phase, len(avail_nid_extracts))
        )
    dfs = []
    if (usecols is not None) and verbose:
        print_log_message("Reading only a subset of the columns")
    for nid, extract_type_id in avail_nid_extracts:
        try:
            df = get_phase_output(
                phase, nid, extract_type_id,
                sub_dirs=sub_dirs,
                exec_function=exec_function,
                exec_function_args=exec_function_args,
                where_filter=where_filter,
                refresh_id=refresh_id,
                usecols=usecols,
                launch_set_id=launch_set_id
            )
        except KeyError as ke:
            fp_signature = (
                f" [phase: {phase}, nid: {nid}, extract_type_id: {extract_type_id}]"
            )
            etext = str(ke) + fp_signature
            if assert_all_available:
                raise KeyError(etext)
            else:
                warnings.warn(etext)
                continue
        if attach_launch_set_id:
            df['launch_set_id'] = get_launch_set_id_of_active(
                phase, nid, extract_type_id, sub_dirs=sub_dirs
            )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=True)
    if verbose:
        print_log_message("Constructed a dataset of {} rows".format(len(df)))

    # filter these again, in case an nid / extract_type_id
    extra_filter_keys = ['location_id', 'year_id']
    for var in extra_filter_keys:
        if var in list(
                dataset_filters.keys()) and dataset_filters[var] is not None:
            vals = dataset_filters[var]
            if not isinstance(vals, list):
                vals = [vals]
            df = df.loc[df[var].isin(vals)]
            if verbose:
                print_log_message(
                    "Pruned dataset by {} filter, leaving "
                    "{} remaining".format(var, len(df))
                )

    return df
