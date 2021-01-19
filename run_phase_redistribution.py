"""
Master script for redistribution.

Creates split groups, sends to run_pipeline_redistribution,
and appends the output from run_pipeline_redistribution.

"""
import sys
import os
import pandas as pd
import numpy as np
import getpass
from mcod_prep.run_phase_redistributionworker import main as worker_main
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from mcod_prep.utils.logging import ymd_timestamp
from cod_prep.downloaders import (
    get_current_location_hierarchy, get_cause_map, get_redistribution_locations,
    add_code_metadata, add_age_metadata
)
from cod_prep.utils import (
    report_if_merge_fail, fill_missing_df, just_keep_trying,
    print_log_message, wait_for_job_ids
)
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import makedirs_safely, get_phase_output, write_phase_output

CONF = Configurator('standard')
USER = getpass.getuser()
CACHE_DIR = CONF.get_directory('db_cache')
CODE_DIR = CONF.get_directory('mcod_code')
LOG_DIR = CONF.get_directory('log_base_dir').format(user=USER)
RD_INPUTS_DIR = CONF.get_directory('rd_process_inputs').format(refresh=CONF.get_id('refresh'))
PACKAGE_DIR = RD_INPUTS_DIR + '/rdp/{csid}'
SG_DIR = CONF.get_directory('rd_process_data') + '/{nid}/{extract_type_id}/{int_cause}/split_{sg}'

# columns specific to redistribution
RD_COLS = ['global', 'dev_status', 'super_region', 'region', 'country', 'subnational_level1',
           'subnational_level2', 'sex', 'age', 'split_group', 'freq', 'site_id']


def convert_deaths_to_died(df, value_col, id_cols):
    """Convert deaths column to died boolean for datasets with two value columns.

    Since redistribution is built to have one 'freq' column, keep 'admissions' as the
    freq column and convert 'deaths' to a boolean 'died'. Essentially do redistribution
    twice for those who have died and those who did not.
    """
    return df.eval(
        f'alive = {value_col} - deaths'
    ).drop(columns=value_col).melt(
        id_vars=id_cols, var_name='died', value_name=value_col
    ).assign(died=lambda d: (d.died == 'deaths') * 1)


def has_garbage(df):
    """Determine whether or not there are any garbage codes."""
    any_garbage = (df['cause_id'] == 743).any()
    return any_garbage


def format_age_groups(df):
    """Convert age groups to simple ages."""
    df = add_age_metadata(df,
                          ['simple_age'],
                          force_rerun=False,
                          block_rerun=True,
                          cache_dir=CACHE_DIR)
    df.rename(columns={'simple_age': 'age'}, inplace=True)
    return df


def drop_zero_deaths(df, value_col='deaths'):
    """Drop rows where there are no deaths."""
    df = df[df[value_col] > 0]
    return df


def add_rd_locations(df, lhh):
    """Merge on location hierarchy specific to redistribution."""
    rd_lhh = get_redistribution_locations(lhh)
    df = pd.merge(df, rd_lhh, on='location_id', how='left')
    report_if_merge_fail(df, 'global', 'location_id')
    report_if_merge_fail(df, 'dev_status', 'location_id')
    return df


def add_split_group_id_column(df, nid, extract_type_id, data_type_id,
                              int_cause, id_col='split_group'):
    """Add group IDs to a dataset.

    Arguments:
    df : DataFrame
         a pandas DataFrame
    group_cols : str or list-like
                 The columns used to group the data
    id_col : str, default 'split_group_id'
             The name of the column where you want to store the group ids
    """
    group_cols = ['country', 'subnational_level1', 'nid', 'extract_type_id', 'year_id', int_cause]
    if 'admissions' in df.columns:
        group_cols += ['died']
    # The aggregator function or column chosen to aggregate doesn't matter
    g = df.groupby(group_cols)[df.columns[-1]].agg(np.min).reset_index()
    # Get rid of all columns except the group columns
    g = g[group_cols]

    # Create ID column that numbers each row (each row is a group)
    g[id_col] = range(1, len(g) + 1)

    return df.merge(g, on=group_cols)


def format_columns_for_rd(df, value_col='deaths'):
    """Ensure necessary columns are appropriately named and present."""
    df.rename(columns={'value': 'cause', value_col: 'freq'}, inplace=True)
    df['sex'] = df['sex_id']

    # don't care about site_id, so set to 2
    df['site_id'] = 2

    missing_cols = []
    for col in RD_COLS:
        if col not in df.columns:
            missing_cols.append(col)
    if len(missing_cols) > 0:
        raise AssertionError(
            "Expected to find ({}) but they were not in "
            "df.columns: ({})".format(missing_cols, df.columns)
        )
    return df


def read_append_split_groups(sg_list, nid, extract_type_id, cause_map, int_cause):
    """Read and append split groups after redistribution.

    Arguments:
    sg_list : list
            a list of all the split groups for given nid
    nid : int
        the nid in the data

    Returns:
        a pandas dataframe of all split groups
        for a given nid appended together
    """
    sg_dfs = []
    for sg in sg_list:
        filepath = (SG_DIR + '/post_rd.csv').format(
            nid=nid, extract_type_id=extract_type_id, int_cause=int_cause, sg=sg)
        sg = just_keep_trying(
            pd.read_csv, args=[filepath], kwargs={'dtype': {'cause': object}}, max_tries=250,
            seconds_between_tries=6, verbose=True)
        sg = merge_acause_and_collapse(sg, cause_map)
        sg_dfs.append(sg)
    df = pd.concat(sg_dfs)
    return df


def revert_variables(df, id_cols, value_col='deaths'):
    """Change things back to standard columns."""
    df.rename(columns={'freq': value_col}, inplace=True)
    if value_col == 'admissions':
        df = df.assign(
            deaths=lambda d: d['died'] * d[value_col]).groupby(
            id_cols, as_index=False)[value_col, 'deaths'].sum()
    else:
        df = df[[value_col] + id_cols]
    return df


def submit_split_group(nid, extract_type_id, split_group, csid, int_cause):
    """Submit jobs by split group."""
    jobname = f"{int_cause}_redistributionworker_{nid}_{extract_type_id}_{split_group}"
    worker = f"{CODE_DIR}/run_phase_redistributionworker.py"
    params = [nid, extract_type_id, split_group, csid, int_cause]
    jid = submit_mcod(jobname, 'python', worker, params=params, cores=1, memory='3G',
                      runtime='00:30:00', verbose=True, logging=True)
    return jid


def write_split_group_input(df, nid, extract_type_id, sg, int_cause):
    """Write completed split group."""
    indir = SG_DIR.format(nid=nid, extract_type_id=extract_type_id, int_cause=int_cause, sg=sg)
    makedirs_safely(indir)
    df.to_csv('{}/for_rd.csv'.format(indir), index=False)


def delete_split_group_output(nid, extract_type_id, sg, int_cause):
    """Delete the existing intermediate split group files."""
    indir = SG_DIR.format(nid=nid, extract_type_id=extract_type_id, int_cause=int_cause, sg=sg)
    for_rd_path = '{}/for_rd.csv'.format(indir)
    post_rd_path = '{}/post_rd.csv'.format(indir)
    for path in [for_rd_path, post_rd_path]:
        if os.path.exists(path):
            os.unlink(path)


def merge_acause_and_collapse(df, cause_map):
    """Add acause column and collapse before appending split groups."""
    cause_map = cause_map[['cause_id', 'value']].copy()
    cause_map = cause_map.rename(columns={'value': 'cause'})
    df = df.merge(cause_map, how='left', on='cause')
    df = df.drop(['cause', 'split_group'], axis=1)
    df = df.groupby([col for col in df.columns if col != 'freq'], as_index=False).sum()
    return df


def run_phase(df, csvid, nid, extract_type_id, lsvid, cmvid, csid, remove_decimal,
              value_col, data_type_id, int_cause, write_diagnostics=True):
    """String together processes for redistribution."""
    # what to do about caching throughout the phase
    read_file_cache_options = {
        'block_rerun': True,
        'cache_dir': CACHE_DIR,
        'force_rerun': False,
        'cache_results': False
    }
    cause_map = get_cause_map(code_map_version_id=cmvid, **read_file_cache_options)
    lhh = get_current_location_hierarchy(location_set_version_id=lsvid, **read_file_cache_options)

    orig_deaths_sum = int(df[value_col].sum())

    print_log_message("Formatting data for redistribution")
    if value_col == 'admissions':
        id_cols = [x for x in df.columns if 'id' in x] + [int_cause]
        df = convert_deaths_to_died(df, value_col, id_cols)
    if remove_decimal:
        print_log_message("Removing decimal from code map")
        cause_map['value'] = cause_map['value'].apply(lambda x: x.replace(".", ""))
    df = add_code_metadata(
        df, ['value', 'code_system_id'], code_map=cause_map, **read_file_cache_options
    )
    df = format_age_groups(df)
    df = drop_zero_deaths(df, value_col=value_col)
    df = add_rd_locations(df, lhh)
    df = fill_missing_df(df, verify_all=True)
    df = add_split_group_id_column(df, nid, extract_type_id, data_type_id, int_cause)

    # final check to make sure we have all the necessary columns
    df = format_columns_for_rd(df, value_col=value_col)

    split_groups = list(df.split_group.unique())
    parallel = len(split_groups) > 1

    print_log_message("Submitting/Running split groups")
    sg_jids = []
    for split_group in split_groups:
        delete_split_group_output(nid, extract_type_id, split_group, int_cause)
        split_df = df.loc[df['split_group'] == split_group]
        write_split_group_input(split_df, nid, extract_type_id, split_group, int_cause)
        if parallel:
            jid = submit_split_group(nid, extract_type_id, split_group, csid, int_cause)
            sg_jids.append(jid)
        else:
            worker_main(nid, extract_type_id, split_group, csid, int_cause)
    if parallel:
        print_log_message("Waiting for splits to complete...")
        wait_for_job_ids(sg_jids, 30)
        print_log_message("Done waiting. Appending them together")
    df = read_append_split_groups(split_groups, nid, extract_type_id, cause_map, int_cause)

    print_log_message("Done appending files - {} rows assembled".format(len(df)))
    post_rd_cols = ['nid', 'extract_type_id', 'location_id', 'year_id', 'sex_id',
                    'cause_id', 'age_group_id', int_cause]
    df = revert_variables(df, post_rd_cols, value_col=value_col)

    after_deaths_sum = int(df[value_col].sum())
    before_after_text = """
        Before GC redistribution: {a}
        After GC redistribution: {b}
    """.format(a=orig_deaths_sum, b=after_deaths_sum)
    diff = abs(orig_deaths_sum - after_deaths_sum)
    # somewhat arbitrary, trying to avoid annoying/non-issue failures
    diff_threshold = max(.02 * orig_deaths_sum, 5)
    if not diff < diff_threshold:
        raise AssertionError("{} not close.\n".format(value_col) + before_after_text)
    else:
        print_log_message(before_after_text)

    return df


def main(nid, extract_type_id, csvid, lsvid, csid, cmvid, remove_decimal, data_type_id, int_cause):
    """Download data, run phase, and output result."""
    df = get_phase_output('format_map', nid, extract_type_id, sub_dirs=int_cause)
    if 'admissions' in df.columns:
        value_col = 'admissions'
    else:
        value_col = 'deaths'

    if has_garbage(df):
        print_log_message("Running redistribution")
        df = run_phase(df, csvid, nid, extract_type_id, lsvid, cmvid, csid,
                       remove_decimal, value_col, data_type_id, int_cause)
    else:
        print_log_message("No redistribution to do.")
        group_cols = list(set(df.columns) - set(['code_id'] + value_col))
        df = df.groupby(group_cols, as_index=False)[value_col].sum()

    # use timestamp as the "launch set id"
    write_phase_output(df, 'redistribution', nid, extract_type_id,
                       ymd_timestamp(), sub_dirs=int_cause)


if __name__ == "__main__":
    nid = int(sys.argv[1])
    extract_type_id = int(sys.argv[2])
    # cause set version id
    csvid = int(sys.argv[3])
    # location_set_version_id
    lsvid = int(sys.argv[4])
    # code_system_id
    csid = int(sys.argv[5])
    # code_map_version_id
    cmvid = int(sys.argv[6])
    # remove decimal
    remove_decimal = sys.argv[7]
    assert remove_decimal in ["True", "False"], \
        "invalid remove_decimal: {}".format(remove_decimal)
    remove_decimal = (remove_decimal == "True")
    data_type_id = int(sys.argv[8])
    int_cause = str(sys.argv[9])
    main(nid, extract_type_id, csvid, lsvid, csid, cmvid,
         remove_decimal, data_type_id, int_cause)
