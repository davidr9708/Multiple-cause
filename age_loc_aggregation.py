"""Aggregate draws of deaths by age and location."""
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
import sys
import pandas as pd
from adding_machine.summarizers import combine_ages
from cod_prep.downloaders import (
    get_current_location_hierarchy, add_location_metadata, add_population, create_age_bins, get_pop,
    get_age_weights
)
from cod_prep.claude.configurator import Configurator
from cod_prep.utils import report_duplicates, report_if_merge_fail, print_log_message

CONF = Configurator('standard')

NUM_DRAWS = 1000
DEM_COLS = ['cause_id', 'location_id', 'sex_id', 'year_id', 'age_group_id']
DRAW_COLS = ["draw_" + str(x) for x in range(0, NUM_DRAWS)]
COD_AGES = CONF.get_id('cod_ages')


def get_lvl2_lvl0_aggregates(df, pop_df=None):
    """Adjust region level + higher locations since GBD does not estimate for every country.

    Basic idea - use population to scale up the deaths.
    """
    # add on population at the most detailed location level
    df = add_population(df, pop_df=pop_df)
    report_if_merge_fail(df, 'population', ['age_group_id', 'sex_id', 'location_id', 'year_id'])

    print_log_message("Creating global aggregates")
    global_df = df.copy()
    global_df['location_id'] = 1
    global_df = scale_location_aggregates(global_df, pop_df)

    print_log_message("Creating region aggregates")
    region_df = df.copy()
    region_df['location_id'] = region_df['region_id']
    region_df = scale_location_aggregates(region_df, pop_df)

    print_log_message("Creating super region aggregates")
    super_region_df = df.copy()
    super_region_df['location_id'] = super_region_df['super_region_id']
    super_region_df = scale_location_aggregates(super_region_df, pop_df)

    scaled_locs_df = pd.concat([global_df, region_df, super_region_df],
                               ignore_index=True, sort=True)

    return scaled_locs_df


def scale_location_aggregates(df, pop_df=None):
    """Adjust region level + higher locations since GBD does not estimate for every country."""
    df = df.copy()
    value_cols = DRAW_COLS + ['population']
    assert 'population' in df.columns

    # collapse and sum up deaths and population
    df = df.groupby(
        ['age_group_id', 'sex_id', 'year_id', 'cause_id', 'location_id'], as_index=False
    )[value_cols].sum()

    # calculate the rate
    for draw in range(0, NUM_DRAWS):
        df['rate_' + str(draw)] = old_div(df['draw_' + str(draw)], df['population'])

    # drop population to merge on the true aggregate population
    df = df.drop('population', axis=1)
    df = add_population(df, pop_df=pop_df)
    report_if_merge_fail(df, 'population', ['age_group_id', 'sex_id', 'location_id', 'year_id'])

    # scale the deaths using the true aggregate population
    for draw in range(0, NUM_DRAWS):
        df['draw_' + str(draw)] = df['rate_' + str(draw)] * df['population']

    # drop added columns
    rate_draws = ["rate_" + str(x) for x in range(0, NUM_DRAWS)]
    df = df.drop(rate_draws + ['population'], axis=1)

    return df


def get_country_aggregate(df, lhh):
    """Aggregate to country level."""
    # use iso3 (from ihme_loc_id) to aggregate
    country_agg_df = add_location_metadata(df, 'ihme_loc_id', location_meta_df=lhh)
    report_if_merge_fail(country_agg_df, 'ihme_loc_id', 'location_id')
    country_agg_df['iso3'] = country_agg_df['ihme_loc_id'].str[0:3]

    # collapse
    group_cols = ['iso3'] + [x for x in DEM_COLS if x != 'location_id']
    country_agg_df = country_agg_df.groupby(group_cols, as_index=False)[DRAW_COLS].sum()

    # get new location_id
    ihme_loc_id_dict = lhh.set_index('ihme_loc_id')['location_id'].to_dict()
    country_agg_df['location_id'] = country_agg_df['iso3'].map(ihme_loc_id_dict)
    country_agg_df.drop('iso3', axis=1, inplace=True)

    return country_agg_df


def aggregate_locs(df, lhh):
    """Create global, country, SDI rows."""
    df = df.copy()

    print_log_message("Reading in population")
    pop_df = get_pop(pop_run_id=CONF.get_id('pop_run'), force_rerun=False, block_rerun=True)
    pop_df = pop_df.query(f'age_group_id in {COD_AGES}')

    data_ages = list(df.age_group_id.unique())
    if len(set(data_ages) - set(COD_AGES)) > 0:
        print_log_message(f"Aggregating population age groups to match: {data_ages} ids")
        pop_df = create_age_bins(pop_df, data_ages, dropna=True)
        pop_df = pop_df.groupby(
            ['age_group_id', 'sex_id', 'location_id', 'year_id'], as_index=False
        )['population'].sum()

    df = add_location_metadata(df, ['region_id', 'super_region_id'], location_meta_df=lhh)
    lvl2_lvl0_df = get_lvl2_lvl0_aggregates(df, pop_df)
    country_df = get_country_aggregate(df, lhh)

    loc_agg_df = pd.concat([lvl2_lvl0_df, country_df], ignore_index=True, sort=True)

    return loc_agg_df


def create_age_standardized_rates(df, draw_cols, dem_cols, location_set=40):
    """Standard function for creating age standardized age groups.

    Arguments
        df - pandas dataframe, must have at least age, sex, location, year columns
        draw_cols - the value columns to collapse
        dem_cols - identifying columns to collapse by
        location_set - always add pop for location_set_id 8 (standard), but
        if there are any missing values, then will add population for this optional location set
    Returns
        a dataframe with only age_group_id 27
    """
    # set block rerun kwargs
    cache_kwargs = {'force_rerun': False, 'block_rerun': True}

    # return a new dataframe, don't alter existing
    df = df.copy()

    # only keep detailed age groups
    df = df.loc[df['age_group_id'].isin(COD_AGES)]

    # pull in the standard population weights
    age_weight_df = get_age_weights(force_rerun=False, block_rerun=True)
    age_weight_dict = age_weight_df.drop_duplicates(
        ['age_group_id', 'age_group_weight_value']
    ).set_index('age_group_id')['age_group_weight_value'].to_dict()
    df['weight'] = df['age_group_id'].map(age_weight_dict)
    report_if_merge_fail(df, 'weight', 'age_group_id')

    # merge on population
    pop_run_id = CONF.get_id('pop_run')
    # default location set is 35
    df = add_population(df, pop_run_id=pop_run_id, **cache_kwargs)
    # if there are still some missing, then add SDI location_ids
    null_pop = df['population'].isnull()
    null_pop_df = df[null_pop].drop('population', axis=1)
    if len(null_pop_df) > 0:
        null_pop_df = add_population(null_pop_df, pop_run_id=pop_run_id,
                                     location_set_id=location_set, **cache_kwargs)
        df = pd.concat([df[~null_pop], null_pop_df], ignore_index=True)
    report_if_merge_fail(df, "population", ['sex_id', 'age_group_id',
                                            'year_id', 'location_id'])

    # age standardized deaths rates = (deaths / population) * weight
    if type(draw_cols) != list:
        draw_cols = [draw_cols]
    for draw_col in draw_cols:
        df[draw_col] = (df[draw_col] / df['population']) * df['weight']

    # then sum up these values
    group_cols = [x for x in dem_cols if x != 'age_group_id']
    df = df.groupby(group_cols, as_index=False)[draw_cols].sum()
    df['age_group_id'] = 27

    return df


def create_agg_age_dict():
    """Create dictionary of {age_group_id: (age_start, age_end)}."""
    # hard coded, not sure how to pull from db: https://hub.ihme.washington.edu/x/iWkCAw
    age_group_ids = [22, 1, 158, 23, 159, 24, 25, 26, 21, 28, 157, 42, 162, 420]
    age_df = get_ages(force_rerun=False, block_rerun=True).query(f'age_group_id in {age_group_ids}')
    return age_df.set_index(
        'age_group_id'
    )[['age_group_years_start', 'age_group_years_end']].apply(tuple, axis=1).to_dict()


def aggregate_sexes(df):
    """Create both sexes rows."""
    df = df.copy()
    group_cols = [x for x in DEM_COLS if x != 'sex_id']
    both_sex_df = df.groupby(group_cols, as_index=False)[DRAW_COLS].sum()
    both_sex_df['sex_id'] = 3
    return both_sex_df


def main(out_dir, cause_id, year_id, end_product):
    df = pd.read_csv("{}/{}_deaths.csv".format(out_dir, cause_id))

    print_log_message("Creating location aggregates")
    lhh = get_current_location_hierarchy(
        location_set_version_id=CONF.get_id('location_set_version'),
        force_rerun=False, block_rerun=True
    )
    loc_agg_df = aggregate_locs(df, lhh)
    # only keep levels that were not created
    df = add_location_metadata(df, 'level', location_meta_df=lhh).query('level > 3')
    df = df.drop('level', axis=1)
    df = pd.concat([df, loc_agg_df], ignore_index=True, sort=True)
    report_duplicates(df, DEM_COLS)

    print_log_message("Creating sex aggregates")
    both_sex_df = aggregate_sexes(df)
    df = pd.concat([both_sex_df, df], ignore_index=True, sort=True)

    print_log_message("Creating age aggregates")
    agg_age_dfs = []
    if end_product in ['mortality', 'incidence']:
        df.assign(measure_id=lambda x: 1 if end_product == 'mortality' else 6)
        agg_age_df = combine_ages(df, gbd_compare_ags=True, metric_id=1,
                                  age_groups=create_agg_age_dict())
        agg_age_dfs.append(agg_age_df)
        DEM_COLS += ['measure_id', 'metric_id']
    # create this separate from central comp function to control pop version
    asr_df = create_age_standardized_rates(df, DRAW_COLS, DEM_COLS)
    df = pd.concat(agg_age_dfs.append(asr_df), ignore_index=True, sort=True)

    # final checks before saving output
    report_duplicates(df, DEM_COLS)
    df[DEM_COLS] = df[DEM_COLS].astype(int)
    df.to_csv(f"{out_dir}/{cause_id}_aggregate.csv", index=False)


if __name__ == "__main__":
    out_dir = str(sys.argv[1])
    cause_id = int(sys.argv[2])
    year_id = int(sys.argv[3])
    end_product = str(sys.argv[4])
    main(out_dir, cause_id, year_id, end_product)
