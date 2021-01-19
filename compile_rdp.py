"""Combine draw files to create redistribution proportions."""
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial

from cod_prep.downloaders import pretty_print, add_age_metadata, add_location_metadata
from cod_prep.claude.configurator import Configurator
from cod_prep.utils import print_log_message, report_duplicates, cod_timestamp
from cod_prep.claude.claude_io import makedirs_safely
from mcod_prep.burden_calculator import BurdenCalculator
from mcod_prep.cause_aggregation import get_cause_list

CONF = Configurator('standard')

DEM_COLS = ['cause_id', 'location_id', 'sex_id', 'year_id', 'age_group_id']
DRAW_COLS = ["draw_" + str(x) for x in range(0, 1000)]
VETTING_DIR = "/snfs1/WORK/03_cod/01_database/mcod/{int_cause}"
# special directory just for making "manual" adjustments to the proportions
# uploading to engine room should be read directly from here
JDRIVE_DIR = "/snfs1/WORK/03_cod/01_database/02_programs_nocode/redistribution/"\
             "regression_proportions/manual_fixes/{int_cause}"


def merge_stars(df):
    stars_df = pd.read_csv("/snfs1/WORK/03_cod/01_database/02_programs_nocode/"
                           "source_metadata/smp/stars_by_iso3_time_window.csv")
    stars_df = stars_df.query(
        'location_level == 3 & time_window == "1980_2017"'
    )[['stars', 'location_id']]
    stars_dict = stars_df.set_index('location_id')['stars'].to_dict()
    df['stars'] = df['location_id'].map(stars_dict)
    return df


def get_location_weights(df, int_cause):
    """Replace the country level proportions with an aggregated location."""
    # first create the dataframe that we'll use to grab the new proportions
    # "cf_mean" is the column holding the value for the proportion
    keep_cols = ['location_id', 'year_id', 'age_group_id', 'sex_id', 'cause_id', 'cf_mean']
    merge_df = df[keep_cols]
    # need a "merge id" for the correct location
    merge_df = merge_df.rename(columns={'location_id': 'merge_id'})

    # subset to only the country level rows, these are what we want to actually use
    # in the garbage packages
    df = df.query('location_level == 3')
    df = add_location_metadata(df, ['region_id', 'super_region_id'])
    # drop the redistribution proportion so we can merge on the correct one
    df = df.drop('cf_mean', axis=1)
    # 4+5 stars will use the country level
    df.loc[df['stars'] >= 4, 'merge_id'] = df['location_id']
    # < 4 stars use the region proportions
    df.loc[df['stars'] < 4, 'merge_id'] = df['region_id']
    # SSA, South Asia will use super region
    df.loc[df['super_region_id'].isin([166, 158]), 'merge_id'] = df['super_region_id']

    # decision made for decomp 2 refresh 2 GBD 2019
    # see /snfs1/WORK/03_cod/01_database/mcod/unsp_hf for documentation
    if int_cause == 'unsp_hf':
        df.loc[df['location_id'] == 196, 'merge_id'] = df['location_id']
        df.loc[df['location_id'] == 67, 'merge_id'] = df['super_region_id']
        df.loc[df['location_id'] == 22, 'merge_id'] = df['super_region_id']
        # more specific changes by year/age for japan
        df.loc[
            (df['location_id'] == 67) & (df['year_id'].isin([1990, 2000])) &
            (df['age_group_id'].isin([47, 21])), 'merge_id'
        ] = 73

    # merge on the correct proportion
    df = df.merge(merge_df, on=['merge_id', 'year_id', 'age_group_id', 'sex_id', 'cause_id'])
    df = df.drop(['region_id', 'super_region_id'], axis=1)

    # create the weight name for that location
    df['wgt_location'] = df['location_name']

    return df


def get_year_weights(df, int_cause):
    df.loc[df['year_id'] == 2015, 'wgt_year'] = '2010-2050'
    df.loc[df['year_id'] == 2010, 'wgt_year'] = '2000-2009'
    df.loc[df['year_id'] == 2000, 'wgt_year'] = '1990-1999'
    df.loc[df['year_id'] == 1990, 'wgt_year'] = '1980-1989'
    # for unsp_hf pkg in japan there is a disjoing in ICD9 vs. ICD10 where ICD9 is years < 1995
    if int_cause == 'unsp_hf':
        df.loc[(df['wgt_location'] == 'Japan') & (df['year_id'] == 2000), 'wgt_year'] = '1990-1994'
        df.loc[(df['wgt_location'] == 'Japan') & (df['year_id'] == 2010), 'wgt_year'] = '1995-2009'
    return df


def get_sex_weights(df):
    df['wgt_sex'] = df['sex_label'].copy(deep=True)
    df['wgt_sex'] = df['wgt_sex'].replace({'male': 'Male', 'female': 'Female',
                                           'both sexes': 'Both sexes'})
    df = add_age_metadata(df, 'age_group_years_end')
    df = df.loc[
        ((df['age_group_years_end'] <= 15) & (df['sex_id'] == 3)) |
        ((df['age_group_years_end'] > 15) & (df['sex_id'].isin([1, 2])))
    ]
    return df


def save_diagnostic_df(df, int_cause):
    """Save country, region, and super region proportions for diagnostics."""
    df = df.copy(deep=True)
    df = df.loc[(df.location_level.isin([1, 2, 3]))]
    # shortens up the dataframe a bit and avoids confusion
    df = get_sex_weights(df)
    dem_cols = ['location_name', 'year_id', 'age_group_name', 'wgt_sex', 'acause',
                'location_level']
    df = df[dem_cols + ['cf_mean']]
    report_duplicates(df, dem_cols)
    df.rename({'cf_mean': 'wgt'}, axis='columns', inplace=True)
    makedirs_safely(VETTING_DIR.format(int_cause=int_cause))
    df.to_excel(
        (VETTING_DIR + '/{int_cause}_diagnostic.xlsx').format(int_cause=int_cause),
        index=False
    )


def adjust_small_weights(df):
    df['adjust'] = 0
    # adjust very small proportions
    df.loc[(df['wgt'] < .0056), 'adjust'] = 1

    # calculate the amount that needs to be moved
    df['move_wgt'] = df['wgt'] * df['adjust']
    df['move_wgt'] = df.groupby('wgt_group_name')['move_wgt'].transform(sum)

    # drop the targets with small weights
    df = df.query('adjust == 0')

    # move the "move_wgt" to the remaining targets, proportional to the weight
    df['total_wgt'] = df.groupby('wgt_group_name')['wgt'].transform(sum)
    df['scalar'] = df['wgt'] / df['total_wgt']
    df['add_wgt'] = df['move_wgt'] * df['scalar']
    df['new_wgt'] = df['add_wgt'] + df['wgt']

    df = df.drop(['move_wgt', 'add_wgt', 'wgt', 'total_wgt'], axis=1)
    df.rename(columns={'new_wgt': 'wgt'}, inplace=True)

    return df


def transform_for_engine_room_upload(df, int_cause):
    """Prep the compiled dataframe to be in the correct format for the engine room."""
    df = merge_stars(df)

    df = get_location_weights(df, int_cause)

    df = get_year_weights(df, int_cause)

    df = get_sex_weights(df)

    # reformat age group name
    df['wgt_age'] = df['age_group_name'].copy(deep=True)
    df['wgt_age'] = df['wgt_age'].replace(
        {' years': '', ' to ': '-', '(Post Neonatal)': '.1-1', '(Neonatal)': '< .1'}, regex=True
    )

    # need wgt_group_name + wgt columns
    df['wgt_group_name'] = df['wgt_location'].str.cat(
        others=[df['wgt_age'], df['wgt_sex'], df['wgt_year']], sep=', '
    )
    df['wgt'] = df['cf_mean'].copy(deep=True)

    df['target_codes'] = df['acause'].copy(deep=True)

    df = adjust_small_weights(df)

    report_duplicates(df, ['wgt_group_name', 'target_codes', 'wgt'])
    wgt_cols = [x for x in df.columns if 'wgt' in x]
    df = df[['target_codes'] + wgt_cols]

    assert np.allclose(df.groupby('wgt_group_name').wgt.sum(), 1)

    return df


def summarize_draws(df, draw_cols=DRAW_COLS, prefix=''):
    """Calculate mean, upper, and lower of draws."""
    df[prefix + 'mean'] = df[draw_cols].mean(axis=1)
    df[prefix + 'upper'] = df[draw_cols].quantile(.975, axis=1)
    df[prefix + 'lower'] = df[draw_cols].quantile(.025, axis=1)
    return df


def prep_sepsis_fraction_tables(year_cause_tuple, parent_dir, subset_string, int_cause):
    """Prep summary tables using all sepsis-related deaths as the denominator.

    Note: These are the fractions that we'll need to use in redistribution.
    """
    print_log_message("Starting year, cause: {}".format(year_cause_tuple))
    year = year_cause_tuple[0]
    cause = year_cause_tuple[1]

    # read in dataframe by year, cause
    indir = os.path.join(parent_dir, str(year))
    df = pd.read_csv("{}/{}_aggregate.csv".format(indir, cause))

    if (cause in [345, 856, 857, 858]) & (int_cause == "hepatic_failure"):
        df = add_age_metadata(df, ['age_group_years_start'], block_rerun=True, force_rerun=False)
        too_old = 10.0 < df['age_group_years_start']
        df = df.loc[~too_old]
        df.drop("age_group_years_start", axis='columns', inplace=True)

    # subset to demographics of interest
    df = df.query("{}".format(subset_string))

    denominator_df = pd.read_csv("{}/_all_int_cause_aggregate.csv".format(indir))
    denominator_df.drop('cause_id', axis=1, inplace=True)

    # merge sepsis-cause deaths and total sepsis-deaths dataframes
    merge_cols = [x for x in DEM_COLS if x != 'cause_id']
    df = df.merge(denominator_df, how='left', on=merge_cols)

    # calculate fractions
    for draw in DRAW_COLS:
        df[draw] = old_div(df["{}_x".format(draw)], df["{}_y".format(draw)])

    # collapse draws of cause fractions
    df = summarize_draws(df, prefix='cf_')
    df = summarize_draws(df, [x for x in df.columns if '_x' in x], prefix='deaths_')
    df = summarize_draws(df, [x for x in df.columns if '_y' in x], prefix='sample_size_')

    df = df[['cf_mean', 'cf_upper', 'cf_lower', 'deaths_mean', 'sample_size_mean'] + DEM_COLS]

    print_log_message("Job done: {}".format(year_cause_tuple))
    return df


def prep_aggregate_results(year_range, cause_list, parent_dir, metric_func,
                           subset_str, filename, int_cause):
    """Show by country, year for all ages, both sexes."""
    print_log_message("Compiling year/cause files")

    # multiprocessing by year/cause
    input_args = [(year, cause_id) for year in year_range for cause_id in cause_list]
    pool = Pool(40)
    assert callable(metric_func)
    _metric_func = partial(metric_func, parent_dir=parent_dir,
                           subset_string=subset_str, int_cause=int_cause)
    df_list = pool.map(_metric_func, input_args)
    pool.close()
    pool.join()

    # append all the files into one dataframe
    df = pd.concat(df_list, ignore_index=True)

    # quick check that everything sums to 1
    assert np.allclose(
        df.groupby(['age_group_id', 'location_id', 'year_id', 'sex_id']).cf_mean.sum(), 1
    )

    # add in 'name' cols for ids
    df = pretty_print(df)

    # Save country, region and super region proportions for diagnostics
    save_diagnostic_df(df, int_cause)

    print_log_message("Formatting for engine room.")
    df = transform_for_engine_room_upload(df, int_cause)

    print_log_message("Saving output")
    # save one to /ihme, this is versioned by the model run and the timestamp
    df.to_csv("{}/{}.csv".format(parent_dir, filename), index=False)
    makedirs_safely(os.path.join(parent_dir, '_archive'))
    df.to_csv(
        os.path.join(parent_dir, '_archive', filename + '_' + cod_timestamp() + '.csv'),
        index=False
    )
    # save file to easily send to Mohsen used for "manual" fixes
    makedirs_safely(JDRIVE_DIR.format(int_cause=int_cause))
    df.to_excel(
        os.path.join(JDRIVE_DIR.format(int_cause=int_cause), 'extracted_weights.xlsx'), index=False
    )


def main(description, int_cause):
    parent_dir = os.path.join(CONF.get_directory('process_data'), int_cause, description)
    # set year range and cause lists to loop over
    # each file is saved in separate year directories by cause
    cause_list = get_cause_list(int_cause, parent_dir)
    year_range = BurdenCalculator.default_year_list
    # compile all cause/year files w/ total intermediate cause deaths as denominator
    prep_aggregate_results(year_range, cause_list, parent_dir, prep_sepsis_fraction_tables,
                           'age_group_id != 22', 'compiled_rd_props', int_cause)


if __name__ == "__main__":
    description = str(sys.argv[1])
    int_cause = str(sys.argv[2])
    main(description, int_cause)
