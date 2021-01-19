"""Prepare cleaned, mapped MCOD outputs for regression and launch model."""

import sys
import re
import pandas as pd
from mcod_prep.utils.covariates import merge_covariate
from mcod_prep.utils.mcause_io import get_mcause_data
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from mcod_prep.utils.causes import get_int_cause_hierarchy
from cod_prep.utils import report_if_merge_fail, print_log_message
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import makedirs_safely

CONF = Configurator('standard')


def fix_sepsis(df, int_cause):
    """Sepsis can have implicit, explicit, or total."""
    sepsis_type = int_cause.split('_')[0]
    # if there's nothing after the '_' then it's total sepsis
    if sepsis_type == 'sepsis':
        df[int_cause] = (df['sepsis'].isin(['explicit', 'implicit'])) * 1
    # otherwise we're modeling explicit_sepsis only, e.g.
    else:
        df[int_cause] = (df['sepsis'] == sepsis_type) * 1


def format_for_model(df, int_cause, end_product):
    """Prep data before modeling."""
    # AGE GROUPS
    keep_ages = CONF.get_id('cod_ages')
    drop_rows = ((df['sex_id'] == 9) | ~(df['age_group_id'].isin(keep_ages)) |
                 (df['cause_id'].isin([919, 744, 743])))
    df = df[~drop_rows]

    # sepsis has implicit and explicit together
    if 'sepsis' in int_cause:
        fix_sepsis(df, int_cause)
    assert int_cause in df.columns, "intermediate cause column is missing!"
    assert set(df[int_cause].unique()) == {0, 1}, \
        "expecting {} column to be 0 or 1".format(int_cause)

    # FORMATTING VALUE COLUMNS
    value_cols = ['successes', 'failures']
    if end_product == 'incidence':
        value_cols += ['cases']
        df['successes'] = (df[int_cause] == 1) * df['deaths']
        df['cases'] = (df[int_cause] == 1) * df['admissions']
        df['failures'] = df['cases'] - df['successes']
    elif end_product == 'mortality':
        value_cols += ['deaths']
        df['successes'] = (df[int_cause] == 1) * df['deaths']
        df['failures'] = df['deaths'] - df['successes']

    # COLLAPSE
    id_cols = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'level_1', 'level_2']
    df = df.groupby(id_cols, as_index=False)[value_cols].sum()

    # FORMATTING FRACTION COLUMN
    if end_product == 'incidence':
        # case fatality rate
        df[int_cause + '_cfr'] = df['successes'] / df['cases']
    elif end_product == 'mortality':
        df[int_cause + '_fraction'] = df['successes'] / df['deaths']

    return df


def merge_nested_cause_levels(df, int_cause):
    """Merge on nesting of cause levels, specific to each intermediate cause."""
    cause_meta_df = get_int_cause_hierarchy(
        int_cause, force_rerun=False, block_rerun=True, cache_dir='standard',
        cause_set_version_id=CONF.get_id('cause_set_version')
    )[['cause_id', 'level_1', 'level_2']]
    df = df.merge(cause_meta_df, how='left', on='cause_id')
    return df


def write_model_input_data(df, int_cause, end_product, description, output_dir):
    """Save input data for model."""
    diagnostic_dir = f'/snfs1/WORK/03_cod/01_database/mcod/{int_cause}/{end_product}/{description}'
    makedirs_safely(diagnostic_dir)
    makedirs_safely(output_dir)
    print_log_message(f"Writing model input file to {output_dir}")
    df.to_csv(f"{output_dir}/model_input.csv", index=False)
    return diagnostic_dir


def launch_modelworker(int_cause, end_product, description, output_dir, diag_dir):
    """Launch two model jobs for explicit and total sepsis."""
    jobname = f'{int_cause}_{end_product}_{description}_modelworker'
    worker = "{}/run_modelworker.R".format(CONF.get_directory('mcod_code'))
    if end_product == 'incidence':
        obs_fraction_col = int_cause + '_cfr'
    elif end_product == 'mortality':
        obs_fraction_col = int_cause + '_fraction'
    params = [int_cause, output_dir, diag_dir, obs_fraction_col]
    log_dir = CONF.get_directory('log_base_dir') + '/model_output'
    submit_mcod(jobname, 'r', worker, 5, '25G', params=params, verbose=True, logging=True,
                jdrive=True, log_base_dir=log_dir, queue='long.q', runtime='05:00:00:00')


def main(description, int_cause, output_dir, end_product):
    data_kwargs = {'phase': 'redistribution', 'sub_dirs': int_cause, 'assert_all_available': True,
                   'force_rerun': True, 'block_rerun': False}
    if re.search('[Tt][eE][sS][tT]', description):
        data_kwargs.update({'iso3': 'ITA'})

    if end_product == 'incidence':
        data_kwargs.update({'data_type_id': 3})
    elif end_product == 'mortality':
        data_kwargs.update({'data_type_id': 9})

    print_log_message("Pulling training data")
    df = get_mcause_data(**data_kwargs)
    df = merge_nested_cause_levels(df, int_cause)
    # drop rows for unknown age/sex, collapse, etc.
    df = format_for_model(df, int_cause, end_product)
    # get the covariates
    covariates_df = pd.read_csv(
        CONF.get_resource('covariates')
    ).query('int_cause == @int_cause')
    assert len(covariates_df) == 1
    covariates = covariates_df['covariates'].str.split(', ').iloc[0]
    for covariate in covariates:
        print_log_message('Merging on {}'.format(covariate))
        df = merge_covariate(df, covariate)

    diag_dir = write_model_input_data(df, int_cause, end_product, description, output_dir)

    print_log_message("Launching model")
    launch_modelworker(int_cause, end_product, description, output_dir, diag_dir)


if __name__ == '__main__':
    description = str(sys.argv[1])
    int_cause = str(sys.argv[2])
    output_dir = str(sys.argv[3])
    end_product = str(sys.argv[4])
    main(description, int_cause, output_dir, end_product)
