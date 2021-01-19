"""Prepare cleaned, mapped MCOD outputs for regression and launch model."""
from __future__ import division

from builtins import str
from past.utils import old_div
import os
import sys
import re
import pandas as pd
from cod_prep.utils import print_log_message
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import makedirs_safely
from cod_prep.downloaders import (
    create_age_bins, add_cause_metadata, add_location_metadata, pretty_print
)
from mcod_prep.utils.causes import get_injuries_cause_ids
from mcod_prep.utils.mcause_io import get_mcause_data
from mcod_prep.utils.covariates import merge_covariate
from mcod_prep.utils.mcod_cluster_tools import submit_mcod

CONF = Configurator('standard')
block_rerun = {'block_rerun': True, 'force_rerun': False}


def format_for_model(df, int_cause, covariates):
    """Prep data before modeling step."""
    # set id_cols and value_cols for collapsing later
    id_cols = ['year_id', 'sex_id', 'age_group_id', 'location_id', 'cause_id']
    value_cols = ['deaths', int_cause + '_deaths', 'non_' + int_cause + '_deaths']

    # bin age groups
    df = create_age_bins(df, get_int_cause_age_bins(int_cause))

    # FORMATTING CAUSES
    # remove secret causes, need to use the estimation cause set version id
    df = add_cause_metadata(
        df, ['parent_id', 'secret_cause', 'yld_only'],
        cause_set_version_id=CONF.get_id('cause_set_version'), **block_rerun
    )
    df.loc[df['secret_cause'] == 1, 'cause_id'] = df['parent_id']
    df = df.drop(['secret_cause', 'parent_id'], axis=1)

    # only keep most detailed causes, need to use the reporting cause set version id
    df = add_cause_metadata(
        df, ['most_detailed', 'parent_id', 'acause'],
        cause_set_version_id=CONF.get_id('reporting_cause_set_version'), **block_rerun
    )
    # make sure that causes where we do not map to the most detailed cause are kept
    lvl3_targets = list(df.loc[(df['most_detailed'] == 0) & (df['deaths'] > 0), 'acause'].unique())
    # others come from the configurator, some overlap here
    lvl3_targets = set(CONF.get_id('level_3_targets') + lvl3_targets)
    start_deaths = df.deaths.sum()
    yld_only_deaths = df.query('yld_only == 1')['deaths'].sum()
    df = df.loc[
        ((df['most_detailed'] == 1) & (df['yld_only'] != 1)) | (df['acause'].isin(lvl3_targets))
    ]
    end_deaths = df.deaths.sum()
    assert start_deaths - end_deaths <= yld_only_deaths, \
        "You've dropped too many deaths, please check."
    # and, of course, there are some exceptions
    for acause in CONF.get_id('level_3_targets'):
        if acause == 'neo_other_benign':
            df.loc[df['acause'].str.startswith('neo_ben_'), 'cause_id'] = df['parent_id']
        else:
            df.loc[df['acause'].str.startswith('{}_'.format(acause)), 'cause_id'] = df['parent_id']

    # FORMATTING VALUE COLUMNS
    # create columns for intermediate cause related deaths and total deaths
    # sepsis has implicit and explicit together
    if 'sepsis' in int_cause:
        sepsis_type = int_cause.split('_')[0]
        if sepsis_type == 'sepsis':
            df[int_cause] = (df['sepsis'].isin(['explicit', 'implicit'])) * 1
        else:
            df[int_cause] = (df['sepsis'] == sepsis_type) * 1
    assert int_cause in df.columns, "intermediate cause column is missing!"
    assert set(df[int_cause].unique()) == {0, 1}, \
        "expecting {} column to be 0 or 1".format(int_cause)
    df[int_cause + '_deaths'] = df[int_cause] * df['deaths']
    df['non_' + int_cause + '_deaths'] = df['deaths'] - df[int_cause + '_deaths']

    # collapse to needed columns
    df = df.groupby(id_cols, as_index=False)[value_cols].sum()

    # calculate observed fractions
    df[int_cause + '_fraction'] = old_div(df[int_cause + '_deaths'], df['deaths'])

    return df


def get_country_level_location_ids(df):
    df = add_location_metadata(df, 'ihme_loc_id',
                               location_set_version_id=CONF.get_id('location_set_version'),
                               **block_rerun)
    df['ihme_loc_id'] = df['ihme_loc_id'].str[0:3]
    df.drop('location_id', axis=1, inplace=True)
    df = add_location_metadata(df, add_cols="location_id", merge_col='ihme_loc_id')
    return df


def format_inj(df, int_cause, timestamp, covariates):
    """Prep injuries data for modeling.
    Use national level because N code patterns are too diverse at the sub national unit
    """
    gdf = pd.read_csv('{}/{}/{}/redistributed_deaths.csv'.
                      format(CONF.get_directory('process_data'), int_cause, timestamp))

    # subset to injuries causes
    inj = get_injuries_cause_ids(int_cause, block_rerun)
    df = df.loc[df.cause_id.isin(inj)]
    # excpetion for this location
    df = df.loc[df.age_group_id != 160]
    df = create_age_bins(df, get_int_cause_age_bins(int_cause))

    # country level df because some injuries not at subnational level
    df = add_location_metadata(df, 'ihme_loc_id')
    df = get_country_level_location_ids(df)
    gdf = get_country_level_location_ids(gdf)

    for covariate in covariates:
        gdf = merge_covariate(gdf, covariate)
        df = merge_covariate(df, covariate)

    id_cols = ['location_id', 'sex_id', 'age_group_id', 'year_id', 'cause_id']
    value_cols = ['{}_deaths'.format(int_cause), 'deaths']
    id_cols += covariates
    cause_names = [x for x in list(gdf) if "inj" in x]

    mdf = pd.melt(gdf, id_vars=id_cols, value_vars=cause_names)
    mdf.drop(['cause_id'], axis=1, inplace=True)
    mdf.rename(columns={'value': '{}_deaths'.format(int_cause), 'variable': 'acause'}, inplace=True)

    # assign underlying cause as target of redistributed garbage deaths
    mdf = add_cause_metadata(mdf, add_cols='cause_id', merge_col='acause',
                             cause_set_version_id=CONF.get_id('reporting_cause_set_version'),
                             **block_rerun)
    cdf = pd.concat([df, mdf], axis=0, ignore_index=True, sort=False)
    cdf[value_cols] = cdf[value_cols].fillna(0)
    cdf = cdf.groupby(id_cols, as_index=False)[value_cols].sum()

    dropping = (cdf['{}_deaths'.format(int_cause)] > cdf['deaths'])
    ddf = cdf[dropping]
    cdf = cdf[~dropping]

    # non int_cause deaths
    cdf['non_' + int_cause + '_deaths'] = cdf['deaths'] - cdf[int_cause + '_deaths']

    # calculate observed fractions
    cdf[int_cause + '_fraction'] = old_div(cdf[int_cause + '_deaths'], cdf['deaths'])

    # drop na's in fraction
    cdf = cdf.loc[cdf[int_cause + '_fraction'].notnull()]

    return cdf, ddf


def get_int_cause_age_bins(int_cause):
    """Return a list of aggregated age group ids.

    These are determined by expert opinion: in which age groups do mortality trends change?
    sepsis: < 28 days, 28 days - 5 years, 5-14, 15-49, 50-74, 75+
    cvd_causes: 0-14, 15-29, 30-44, 45-59, 60-69, 70-79, 80+
    inj: 0-14, 15-49, 50-59, 60-69, 70-79, 80-89, 90+
    """
    cvd_causes = ['pulmonary_embolism', 'right_hf', 'left_hf', 'unsp_hf', 'arterial_embolism']
    cvd = [(x, [39, 195, 211, 222, 229, 47, 21]) for x in cvd_causes + ['aki']]
    inj = [(x, [39, 24, 224, 229, 47, 268, 294]) for x in ['x59', 'y34']]
    sepsis = [
        (x, [28, 5, 23, 24, 41, 234]) for x in ['sepsis', 'explicit_sepsis', 'hepatic_failure']
    ]
    resp = [(x, [28, 5, 23, 24, 224, 229, 47, 30, 160]) for x in ['arf', 'pneumonitis', 'unsp_cns']]
    int_cause_age_bin_dict = dict(cvd + inj + sepsis + resp)
    return int_cause_age_bin_dict[int_cause]


def write_model_input_data(df, timestamp, int_cause, output_dir):
    """Save input data for model."""
    diagnostic_dir = f'/snfs1/WORK/03_cod/01_database/mcod/{int_cause}/rdp/{timestamp}'
    makedirs_safely(diagnostic_dir)
    makedirs_safely(output_dir)
    print_log_message("Writing intermediate cause model input file")
    df.to_csv("{}/model_input.csv".format(output_dir), index=False)
    (top_df, top_pct) = get_causes_for_most_deaths(df, int_cause)
    print_log_message("Writing causes for top {} percent of deaths".format(top_pct))
    top_df.to_csv("{}/top_{}_causes.csv".format(output_dir, top_pct), index=False)
    return diagnostic_dir


# def get_lambda(int_cause):
#     """Get pre-selected values of lambda, based on returning 95% of deaths.

#     Used cv.glmnet to get values of lambda that returns a cause list representing
#     95% of the deaths in the input data. "get_most_deaths.py" determines the number
#     of causes to aim for.
#     """
#     lambda_int_cause_dict = {
#         'explicit_sepsis': 0.0015, 'aki': 0.0005, 'pulmonary_embolism': 0.00022,
#         'arterial_embolism': .000035, 'left_hf': .0011, 'right_hf': .00012, 'unsp_hf': 0.00055
#     }
#     return lambda_int_cause_dict[int_cause]


def get_causes_for_most_deaths(df, int_cause):
    """Return the list of causes comprising 80% of intermediate cause related deaths."""
    df = df.groupby(['cause_id'], as_index=False)[int_cause + '_deaths'].sum()
    df['total'] = df[int_cause + '_deaths'].sum()
    df = df.sort_values(int_cause + '_deaths', ascending=False).reset_index(drop=True)

    top_pct = 80
    # Exception for hepatic_failure
    if int_cause == 'hepatic_failure':
        top_pct = 90

    cutoff = float(top_pct / 100.0) * df.total.loc[0]
    df['cumsum'] = df[int_cause + '_deaths'].cumsum()
    df = df.loc[df['cumsum'] <= cutoff]
    return df, top_pct


def run_model(df, timestamp, int_cause, output_dir):
    """Run model."""
    diag_dir = write_model_input_data(df, timestamp, int_cause)
    jobname = int_cause + "_modelworker_" + timestamp
    worker = os.path.join(CONF.get_directory('mcod_code'), "run_modelworker.R")
    makedirs_safely(output_dir)
    params = [int_cause, output_dir, diag_dir]
    log_base_dir = CONF.get_directory('log_base_dir') + 'model_output'
    submit_mcod(jobname, 'r', worker, 5, '25G', params=params, verbose=True,
                logging=True, jdrive=True, log_base_dir=log_base_dir)


def main(description, int_cause, output_dir):
    data_kwargs = {'phase': 'format_map', 'sub_dirs': int_cause,
                   'data_type_id': 9, 'assert_all_available': True}
    if re.search('[Tt][eE][sS][tT]', description):
        print_log_message("THIS IS A TEST!")
        data_kwargs.update({'iso3': 'ITA'})
        description = "TEST"
    if 'sepsis' in int_cause:
        # explicit sepsis, implicit sepsis, and total sepsis mapped together in one file
        data_kwargs.update({'sub_dirs': 'sepsis'})
    elif int_cause == 'left_hf':
        # for left_hf, there is a shift in coding from icd9 to icd10, but just in the US
        # we only have US data for icd9, so just keep icd10 for this model
        data_kwargs.update({'code_system_id': 1})

    print_log_message("Pulling training data")
    df = get_mcause_data(**data_kwargs)

    # if this wasn't done in the format/map phase, drop unbelievable deaths
    if 'drop_rows' in df.columns:
        assert df.drop_rows.isin([0, 1]).all()
        df = df.loc[df.drop_rows == 0]

    # drop rows for unknown age/sex, cc_code, still births, garbage codes
    drop_rows = ((df['sex_id'] == 9) | (df['age_group_id'] == 283) |
                 (df['cause_id'].isin([919, 744, 743])))
    df = df[~drop_rows]

    # get the covariates for each intermediate cause model
    covariates_df = pd.read_csv(
        CONF.get_resource('covariates')
    ).query('int_cause == @int_cause')
    assert len(covariates_df) == 1
    covariates = covariates_df['covariates'].str.split(', ').iloc[0]

    if int_cause not in ['x59', 'y34']:
        df = format_for_model(df, int_cause, covariates)
        for covariate in covariates:
            print_log_message('Merging on {}'.format(covariate))
            df = merge_covariate(df, covariate)
    else:
        df, ddf = format_inj(df, int_cause, description, covariates)
        ddf = pretty_print(ddf)
        ddf.to_csv('{}/{}/drop_patterns/national_{}_{}.csv'.format(
            CONF.get_directory('mcod_dir'), int_cause, int_cause, description), index=False)

    run_model(df, description, int_cause, output_dir)


if __name__ == '__main__':
    description = str(sys.argv[1])
    int_cause = str(sys.argv[2])
    output_dir = str(sys.argv[3])
    main(description, int_cause, output_dir)
