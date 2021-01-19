from __future__ import division
from builtins import range
from past.utils import old_div
import pandas as pd
import numpy as np
import itertools
from sklearn import preprocessing
from db_tools import ezfuncs
from db_queries import get_covariate_estimates, get_demographics, get_best_model_versions
from mcod_prep.utils.causes import get_int_cause_hierarchy
from cod_prep.claude.configurator import Configurator
from cod_prep.utils import (
    report_if_merge_fail, report_duplicates, enforce_sex_restrictions, enforce_asr,
    get_function_results, print_log_message
)
from cod_prep.downloaders import (
    get_current_cause_hierarchy, get_ages, add_cause_metadata, add_age_metadata,
    add_location_metadata, get_current_location_hierarchy, create_age_bins, add_population
)

CONF = Configurator('standard')


def get_covariate_id(covariate):
    """Pull the covariate_id from shared.covariate using covariate_name_short."""
    query = """SELECT * FROM shared.covariate WHERE covariate_name_short = '{}'""".format(covariate)
    df = ezfuncs.query(query, conn_def=CONF.get_database_setup('db'))
    assert len(df) == 1
    return df['covariate_id'].iloc[0]


def get_covariate_id_cols(covariate_id):
    """Determine identification columns for a given covariate."""
    query = """SELECT * FROM shared.covariate WHERE covariate_id = {}""".format(covariate_id)
    df = ezfuncs.query(query, conn_def=CONF.get_database_setup('db'))
    assert len(df) == 1
    id_cols = ['location_id', 'year_id']
    if df['by_age'].iloc[0] == 1:
        id_cols += ['age_group_id']
    if df['by_sex'].iloc[0] == 1:
        id_cols += ['sex_id']
    return id_cols


def add_weight_to_covariate(cov_df, covariate_name_short, **cache_kwargs):
    """
    Add a weight to a covariate for aggregation.

    Some covariates must be aggregated using a weight. This function chooses the
    appropriate weight based on the covariate, then merges the weight in.

    Arguments:
        cov_df (pd.DataFrame): dataframe of covariates
        covariate_name_short (string): covariate_name_short of cov_df

    Returns:
        cov_df with weight column added
    """
    # Get covariate name if not provided
    if covariate_name_short is None:
        covariate_name_short = cov_df.loc[0, 'covariate_name_short']

    # Choose and add the weights for the covariate
    # Right now population and no weight are the only options
    cov_to_weight_map = {'alcohol_lpc': 'population'}
    weight = cov_to_weight_map[covariate_name_short]

    if weight == 'none':
        cov_df['weight'] = 1
    elif weight == 'population':
        cov_df = add_population(cov_df, pop_run_id=CONF.get_id('pop_run'),
                                location_set_id=CONF.get_id('location_set'),
                                decomp_step=CONF.get_id('decomp_step'), **cache_kwargs)
        report_if_merge_fail(cov_df, check_col='population',
                             merge_cols=['age_group_id', 'location_id', 'year_id', 'sex_id'])
        cov_df.rename({'population': 'weight'}, axis='columns', inplace=True)
    else:
        raise NotImplementedError

    return cov_df


def aggregate_covariate_by_age(cov_df, covariate_name_short=None, agg_age_group_ids=None,
                               id_cols=None, **cache_kwargs):
    """
    Aggregate a covariate by age using a weighted average or weighted sum.

    The weights are chosen based on the covariate.

    Arguments:
        cov_df (pd.DataFrame): dataframe of covariates with at least columns id_cols + 'mean_value'
        covariate_name_short (string): covariate_name_short of cov_df
        agg_age_group_ids (list): list of age_group_ids to aggregate to
        id_cols (list): list of id columns for the covariate

    Returns:
        cov_df with only columns id_cols + 'mean_value' present, aggregated to the age groups
        in agg_age_group_ids using a weighted average or sum
    """
    # Get inputs if not provided
    if covariate_name_short is None:
        covariate_name_short = cov_df.loc[0, 'covariate_name_short']
    if id_cols is None:
        covariate_id = get_covariate_id(covariate_name_short)
        id_cols = get_covariate_id_cols(covariate_id)

    # Add weight column
    cov_df = add_weight_to_covariate(cov_df, covariate_name_short, **cache_kwargs)

    # Multiply the covariate by the weights
    cov_df['weighted_cov'] = cov_df['mean_value'] * cov_df['weight']

    # Map ages to aggregate ages
    cov_df = create_age_bins(cov_df, agg_age_group_ids, dropna=False)

    # Aggregate ages and sum weighted means and weights
    cov_df = cov_df.groupby(id_cols, as_index=False)[['weighted_cov', 'weight']].sum()

    # Choose a sum or an average and aggregate the covariate
    cov_to_agg_type_map = {'alcohol_lpc': 'average'}
    agg_type = cov_to_agg_type_map[covariate_name_short]

    if agg_type == 'average':
        cov_df['agg_cov'] = old_div(cov_df['weighted_cov'], cov_df['weight'])
    elif agg_type == 'sum':
        cov_df['agg_cov'] = cov_df['weighted_cov']
    else:
        raise NotImplementedError

    cov_df.drop(['weighted_cov', 'weight'], axis='columns', inplace=True)
    cov_df.rename({'agg_cov': 'mean_value'}, axis='columns', inplace=True)
    assert set(cov_df.columns) == set(id_cols + ['mean_value'])
    assert cov_df.notnull().values.all()

    return cov_df


def merge_covariate(df, covariate_name_short, scale=False, **get_cov_kwargs):
    """Merge a covariate onto the main dataframe.

    Use covariate_name_short so it lines up with the shared.covariate table.
    """
    covariate_id = get_covariate_id(covariate_name_short)
    id_cols = get_covariate_id_cols(covariate_id)
    cov_df = get_cov(covariate_id=covariate_id, **get_cov_kwargs)[id_cols + ['mean_value']]

    # check if the covariate needs to be aggregated by age
    if 'age_group_id' in id_cols:
        df_age_group_ids = df.age_group_id.unique().tolist()
        cov_age_group_ids = cov_df.age_group_id.unique().tolist()

        if not set(df_age_group_ids).issubset(set(cov_age_group_ids)):
            cache_kwargs = {'force_rerun': False, 'block_rerun': True}
            print_log_message("Aggregating covariates to match incoming dataframe.")
            cov_df = add_age_metadata(
                cov_df, ['age_group_days_start', 'age_group_days_end'], **cache_kwargs
            )
            df_ag = add_age_metadata(
                df.copy(), ['age_group_days_start', 'age_group_days_end'], **cache_kwargs
            )
            too_young = cov_df['age_group_days_end'] <= df_ag.age_group_days_start.min()
            too_old = cov_df['age_group_days_start'] >= df_ag.age_group_days_end.max()
            cov_df = cov_df[~(too_young | too_old)]
            cov_df = cov_df.drop(['age_group_days_start', 'age_group_days_end'], axis=1)

            cov_df = aggregate_covariate_by_age(
                cov_df, covariate_name_short=covariate_name_short,
                agg_age_group_ids=df_age_group_ids, id_cols=id_cols, **cache_kwargs
            )

    cov_df = cov_df.rename(columns={'mean_value': covariate_name_short})
    report_duplicates(cov_df, id_cols)
    if covariate_name_short == 'haqi':
        print_log_message('scaling HAQ index to be between 0 and 1')
        cov_df[covariate_name_short] = old_div(cov_df[covariate_name_short], 100)
    if covariate_name_short == 'LDI_pc':
        print_log_message('using natural log of LDI per capita')
        cov_df[covariate_name_short] = np.log(cov_df[covariate_name_short])

    if scale:
        # helpful! http://benalexkeen.com/feature-scaling-with-scikit-learn/
        print_log_message("Scaling all covariates to be between 0 and 1")
        scaler = preprocessing.MinMaxScaler()
        cov_df[[covariate_name_short]] = scaler.fit_transform(cov_df[[covariate_name_short]])

    df = df.merge(cov_df, on=id_cols, how='left')
    report_if_merge_fail(df, covariate_name_short, id_cols)

    return df


def enforce_asr_aggregate_ages(df, cause_meta_df, age_meta_df):
    """Enforce age/sex restrictions when you're working with aggregate age groups.

    Only slightly more complicated than the enforce_asr used for cod prep.
    """
    df = add_cause_metadata(
        df, add_cols=['yll_age_start', 'yll_age_end', 'male', 'female', 'yld_only'],
        cause_meta_df=cause_meta_df
    )
    df = add_age_metadata(
        df, add_cols=['age_group_years_start', 'age_group_years_end'],
        age_meta_df=age_meta_df
    )
    df = enforce_sex_restrictions(df)
    # inclusive, 'yll_age_start' is start of the age interval
    age_violation_1 = (df['yll_age_start'] >= df['age_group_years_end'])
    # exclusive, 'yll_age_end' is the start of the age interval
    age_violation_2 = (df['yll_age_end'] < df['age_group_years_start'])
    df = df[~(age_violation_1 | age_violation_2 | (df['yld_only'] == 1))]
    df.drop(['yll_age_start', 'yll_age_end', 'male', 'female', 'yld_only',
             'age_group_years_start', 'age_group_years_end'], axis=1, inplace=True)
    return df


def validate_redistribution_restrictions(df, cause_meta_df):
    """Validate redistribution overrides."""
    valid_ages = [0, 0.01, 0.1, 1] + list(range(5, 95, 5))
    bad_age_starts = set(df.age_start.dropna()) - set(valid_ages)
    assert bad_age_starts == set(
    ), "Age starts in redstribution restrictions are invalid {}".format(bad_age_starts)
    bad_age_ends = set(df.age_start.dropna()) - set(valid_ages)
    assert bad_age_ends == set(), \
        "Age ends in redstribution restrictions are invalid {}".format(bad_age_ends)
    valid_locations = set(get_current_location_hierarchy(
        location_set_version_id=CONF.get_id('location_set_version'),
        force_rerun=False, block_rerun=True
    ).location_name).union(set(["NONE"]))
    all_locations = set(df.super_region.dropna()).union(
        set(df.region.dropna())).union(
        set(df.country.dropna())).union(
        set(df.subnational_level1.dropna())).union(
        set(df.subnational_level2.dropna()))
    bad_locations = set(all_locations) - set(valid_locations)
    assert bad_locations == set(
    ), "Locations in redstribution restrictions are invalid {}".format(bad_locations)
    valid_years = list(range(1900, 2100))
    bad_years = set(df.year_start.dropna()).union(
        set(df.year_end.dropna())) - set(valid_years)
    assert bad_years == set(), \
        "years in redstribution restrictions are invalid {}".format(bad_years)
    valid_dev_status = ["D0", "D1"]
    bad_dev_status = set(df.dev_status.dropna()) - set(valid_dev_status)
    assert bad_dev_status == set(
    ), "Dev_status in redstribution restrictions are invalid {}".format(bad_dev_status)
    bad_causes = set(df.acause) - set(cause_meta_df.acause)
    assert bad_causes == set(
    ), "Acauses in redstribution restrictions don't exist " \
        "in cause hierarchy {}".format(bad_causes)


def enforce_redistribution_restrictions(int_cause, cause_meta_df, age_meta_df, template):
    """Prep restrictions on redstribution targets."""
    if int_cause not in ['x59', 'y34']:
        df = pd.read_csv(CONF.get_resource('redistribution_logic_overrides'))
    else:
        df = pd.read_csv(
            CONF.get_resource('injuries_logic_overrides'), dtype={'super_region': 'object'})
    validate_redistribution_restrictions(df, cause_meta_df)
    df['super_region'] = df['super_region'].replace({'NONE': None})
    # this function was written quickly to handle existing restrictions
    # if there are more in the future, then add those in, but fail loudly
    assert df[
        ['region', 'country', 'subnational_level2', 'subnational_level1', 'year_start',
         'year_end', 'sex', 'dev_status']
    ].isnull().values.all(), 'write out other needed restrictions'
    df = df[['acause', 'age_start', 'age_end', 'super_region']]
    df['age_start'] = df['age_start'].fillna(0)
    df['age_end'] = df['age_end'].fillna(95)
    # these are restrictions that we do not want to enforce
    ignore_acauses = CONF.get_id('ignore_redistribution_overrides')
    df = df.loc[~(df['acause'].isin(ignore_acauses))]
    # add acause to template to merge on restrictions
    df = add_cause_metadata(df, add_cols='cause_id', merge_col='acause',
                            cause_meta_df=cause_meta_df)
    orig_cols = template.columns
    template = template.merge(df, on='cause_id', how='left')
    # if nothing merges, then there are no restrictions to apply
    if not template.acause.isnull().values.all():
        template = add_location_metadata(
            template, location_set_version_id=CONF.get_id('location_set_version'),
            force_rerun=False, block_rerun=True, add_cols='super_region_name'
        )
        template = add_age_metadata(
            template, add_cols=['age_group_years_start', 'age_group_years_end'],
            age_meta_df=age_meta_df
        )
        # inclusive, 'age_start' is start of the age interval
        too_young = template['age_start'] >= template['age_group_years_end']
        # exclusive, 'yll_age_end' is the start of the age interval
        too_old = template['age_end'] < template['age_group_years_start']
        template['super_region'] = template['super_region'].fillna(template['super_region_name'])
        loc_violation = template['super_region_name'] != template['super_region']
        template = template[~(too_young | too_old | loc_violation)]
    template = template[orig_cols]
    return template


def prep_predictions_template(year_id, cause_id, out_dir, int_cause, model_input_df, end_product):
    """Prep predictions template from model by year/cause."""
    # first get year/cause/age/sex/location together
    dem_dict = get_demographics(gbd_team="cod", gbd_round_id=CONF.get_id('gbd_round'))
    model_ages = list(model_input_df['age_group_id'].unique())
    dem_dict.update({'cause_id': [cause_id], 'year_id': [year_id], 'age_group_id': model_ages})

    # create a "square" dataframe of all possible year/cause/age/sex combinations
    rows = itertools.product(*list(dem_dict.values()))
    template = pd.DataFrame.from_records(rows, columns=list(dem_dict.keys()))

    # don't create rows that violate age/sex restrictions
    kwargs = {'force_rerun': False, 'block_rerun': True,
              'cause_set_version_id': CONF.get_id('cause_set_version')}
    cause_meta_df = get_current_cause_hierarchy(**kwargs)
    age_meta_df = get_ages(force_rerun=False)
    detail_ages = CONF.get_id('cod_ages')
    if len(set(model_ages) - set(detail_ages)) == 0:
        template = enforce_asr(template, cause_meta_df, age_meta_df)
    else:
        template = enforce_asr_aggregate_ages(template, cause_meta_df, age_meta_df)

    # enforce special restrictions for creating redistribution proportions
    if end_product == 'rdp':
        template = enforce_redistribution_restrictions(
            int_cause, cause_meta_df, age_meta_df, template)

    if ('sepsis' in int_cause) & (end_product in ['mortality', 'incidence']):
        # need to merge on level_1 and level_2 instead of cause_id
        custom_ch = get_int_cause_hierarchy(
            int_cause, **kwargs).set_index('level_2')['level_1'].to_dict()
        template['level_1'] = template['cause_id'].map(custom_ch)
        template['level_2'] = template['cause_id'].copy()
        template.drop('cause_id', axis=1, inplace=True)

    # get the covariates for each intermediate cause model
    covariates_df = pd.read_csv(
        CONF.get_resource('covariates')
    ).query('int_cause == @int_cause')
    assert len(covariates_df) == 1
    covariates = covariates_df['covariates'].str.split(', ').iloc[0]
    for covariate in covariates:
        template = merge_covariate(template, covariate)

    template.to_csv('{}/{}_template.csv'.format(out_dir, cause_id), index=False)


def get_cov(covariate_id=None, model_version_id=None, location_set_id=None,
            gbd_round_id=None, decomp_step=None, **cache_kwargs):
    """Pull covariates with optional caching."""

    # Get inputs if not provided
    if location_set_id is None:
        location_set_id = CONF.get_id('location_set')
    if gbd_round_id is None:
        gbd_round_id = CONF.get_id('gbd_round')
    if decomp_step is None:
        decomp_step = CONF.get_id('decomp_step')
    if model_version_id is None:
        model_version_id = get_best_model_versions(
            entity='covariate', ids=covariate_id, gbd_round_id=gbd_round_id,
            decomp_step=decomp_step, status='best'
        ).loc[0, 'model_version_id']

    cache_name = "cov_{}_mvid_{}_lsid_{}".format(covariate_id, model_version_id, location_set_id)
    function = get_covariate_estimates
    args = [covariate_id]

    kwargs = {
        'location_set_id': location_set_id,
        'gbd_round_id': gbd_round_id,
        'decomp_step': decomp_step,
        'model_version_id': model_version_id
    }

    df = get_function_results(
        function,
        args,
        kwargs,
        cache_name,
        **cache_kwargs
    )

    # Validate model version against covariate
    assert covariate_id == df.loc[0, 'covariate_id'],\
        "Covariate {} and model version {} do not match.".format(covariate_id, model_version_id)

    return df
