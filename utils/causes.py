import pandas as pd
from db_queries import get_cause_metadata
from cod_prep.claude.configurator import Configurator
from cod_prep.utils import (
    report_if_merge_fail, get_function_results, report_duplicates
)
from cod_prep.downloaders import (
    add_cause_metadata, get_best_cause_hierarchy_version,
    get_current_cause_hierarchy, get_cod_ages, get_age_weights, add_population
)

CONF = Configurator('standard')


def _get_sepsis_cause_hierarchy(int_cause, cause_set_version_id):
    """Prep csv with sepsis cause hierarchy.

    Returns a dataframe with columns for level_1 (normal parent_id), level_2
    (special sepsis levels), and cause_id. This hierarchy is used for modeling
    sepsis with level_2 nested within level_1.
    """
    assert 'sepsis' in int_cause, "this hierarchy is just for sepsis"
    df = pd.read_excel(
        "/snfs1/WORK/00_dimensions/03_causes/Copy of new hairarchy-5.xlsx",
        header=1
    )
    # first create a dictionary of the nested levels: names for later
    nested_name_dict = df[
        ['Nested code ', 'acause']
    ].dropna().drop_duplicates().set_index('Nested code ')['acause'].to_dict()

    # drop these rows, they are for new levels
    df = df.loc[df['level'] != 'XX']
    # this row is a duplicate, drop old version (new in green)
    df = df.loc[~((df['cause_id'] == 605) & (df['parent nested code'] == 2400))]
    df['cause_id'] = df['cause_id'].replace({371: 995, 488: 490})
    # fix null nested codes
    df['parent nested code'] = df['parent nested code'].fillna(df['Nested code '])
    assert not df['cause_id'].duplicated().values.any()

    # prep cause hierarchy
    chh = get_current_cause_hierarchy(
        cause_set_version_id=cause_set_version_id, force_rerun=False, block_rerun=True
    )
    # we only want to most detailed causes, except for the secret causes
    # where we also want to model the parent
    chh = chh.loc[
        (chh['yld_only'] != 1) & (chh['parent_id'] != 294) & (chh['is_estimate'] == 1)
    ]
    # level_2 is the cause_id, unless it's a secret cause
    chh['level_2'] = chh['cause_id']
    chh.loc[chh['secret_cause'] == 1, 'level_2'] = chh['parent_id']

    level_dict = df.set_index('cause_id')['parent nested code'].to_dict()
    chh['level_1'] = chh['cause_id'].map(level_dict)

    # add in new causes for GBD 2019
    chh.loc[chh['cause_id'] == 1003, 'level_1'] = 1300
    chh.loc[chh['cause_id'] == 1004, 'level_1'] = 1800
    chh.loc[chh['cause_id'].isin([1018, 1019]), 'level_1'] = 1000
    chh.loc[chh['cause_id'] == 1020, 'level_1'] = 1400
    chh.loc[chh['cause_id'].isin([1006, 1007]), 'level_1'] = 1700
    chh.loc[chh['cause_id'].isin([1005] + list(range(1008, 1014))), 'level_1'] = 1600
    report_if_merge_fail(chh, 'level_1', 'cause_id')
    chh['level_1'] = chh['level_1'].astype(int)

    # overrides for secret causes; need same level_1 as their parent
    chh.loc[chh['cause_id'].isin([343, 407, 1003, 960]), 'level_1'] = 1000
    chh.loc[chh['cause_id'] == 505, 'level_1'] = 1800
    chh.loc[chh['cause_id'] == 720, 'level_1'] = 2600
    chh.loc[chh['cause_id'] == 940, 'level_1'] = 2500
    # check that there are no level_2 rows with different level_1
    check_df = chh.drop_duplicates(['level_1', 'level_2'])
    report_duplicates(check_df, 'level_2')

    # keep names for nesting levels
    chh['nested_name'] = chh['level_1'].map(nested_name_dict)
    report_if_merge_fail(chh, 'nested_name', 'level_1')

    chh = chh[['cause_id', 'level_1', 'level_2', 'acause', 'nested_name']].drop_duplicates()

    return chh


def _get_int_cause_hierarchy(int_cause, cause_set_version_id):
    """Prep csv with non-sepsis nested modeling hierarchy.

    Returns a dataframe with columns for level_1, level_2, and the cause_id.
    level_1: the nested category (should be ~10-15 broad categories)
    level_2: the cause_id (unless it's a secret cause)
    cause_id: cause_id according to the GBD estimation cause set
    """
    chh = get_current_cause_hierarchy(
        cause_set_version_id=cause_set_version_id, force_rerun=False, block_rerun=True
    )
    chh['level_2'] = chh['cause_id']
    chh.loc[chh['secret_cause'] == 1, 'level_2'] = chh['parent_id']
    # set level_1 to the level2 parent in the cause hierarchy
    path_to_parent_df = chh['path_to_top_parent'].str.split(',', expand=True)
    chh['level_1'] = path_to_parent_df[2]
    assert chh.loc[chh['level_1'].isnull(), 'level'].isin([0, 1]).values.all()
    chh = chh.loc[chh['level'] >= 2]
    chh = chh[['cause_id', 'level_1', 'level_2', 'acause']].drop_duplicates()
    return chh


def get_int_cause_hierarchy(int_cause, cause_set_id=4, gbd_round_id=None,
                            cause_set_version_id=None, **cache_kwargs):
    """Get the nested cause hierarchy for modeling intermediate causes."""
    if not gbd_round_id:
        gbd_round_id = CONF.get_id('gbd_round')

    if not cause_set_version_id:
        # need cause_set_version_id to name and search for the cached file
        cause_set_version_id = get_best_cause_hierarchy_version(
            cause_set_id, gbd_round_id)

    if 'sepsis' in int_cause:
        function = _get_sepsis_cause_hierarchy
    else:
        function = _get_int_cause_hierarchy
    cache_name = f"{int_cause}_nested_cause_hierarchy_v{cause_set_version_id}"

    args = [int_cause, cause_set_version_id]
    kwargs = {}

    df = get_function_results(
        function,
        args,
        kwargs,
        cache_name,
        **cache_kwargs
    )

    return df


def get_level_cause_dict(cache_kwargs, intermediate_cause,
                         cause_set_version_id=None,
                         cause_set_id=3, gbd_round_id=None):
    """
    Return a dictionary of cause levels as keys and
    values as a list of cause_ids.

    Default cause_set_id is for the GBD reporting hierarchy
    """
    cause_df = get_current_cause_hierarchy(cause_set_version_id=None,
                                           cause_set_id=3, gbd_round_id=None,
                                           **cache_kwargs)
    cause_df = cause_df.loc[cause_df['yld_only'] != 1]
    no_rd_acauses = get_redistribution_restricted_acauses()
    rd_acause_rows = cause_df['acause'].isin(no_rd_acauses)
    no_rd_parents = list(cause_df.loc[rd_acause_rows, 'parent_id'].unique())
    cause_df = cause_df.loc[~(rd_acause_rows)]
    bad_parents = []
    for parent_id in no_rd_parents:
        check_df = cause_df.loc[cause_df['parent_id'] == parent_id]
        if len(check_df) == 0:
            bad_parents += [parent_id]
    cause_df = cause_df.loc[~(cause_df['cause_id'].isin(bad_parents))]
    cause_df = cause_df.loc[cause_df['most_detailed'] == 0]
    level_id_dict = cause_df.groupby('level')['cause_id'].apply(list).to_dict()
    return level_id_dict


def get_redistribution_restricted_acauses():
    df = pd.read_csv(CONF.get_resource('redistribution_logic_overrides'))
    meta_cols = set(df.columns) - set(['acause', 'note'])
    no_rd_acauses = df[df[list(meta_cols)].isnull(
    ).values.all(axis=1)]['acause'].unique()

    # these are restrictions that we do not want to enforce
    ignore_acauses = CONF.get_id('ignore_redistribution_overrides')
    no_rd_acauses = set(no_rd_acauses) - set(ignore_acauses)

    return list(no_rd_acauses)


def get_detailed_target_list(cause_set_version_id=None, cause_set_id=3,
                             gbd_round_id=None, **cache_kwargs):
    """Pull a list of most detailed from GBD reporting hierarchy.

    These are the targets for redistribution,
    we also want to drop restricted causes.
    """
    cause_df = get_current_cause_hierarchy(cause_set_version_id=None,
                                           cause_set_id=3, gbd_round_id=None,
                                           **cache_kwargs)
    cause_df = cause_df.loc[(cause_df['yld_only'] != 1) & (
        cause_df['most_detailed'] == 1)]

    # drop causes that we block entirely from redistribution
    no_rd_acauses = get_redistribution_restricted_acauses()

    # drop rows that we do not want to choose as redistribution targets
    cause_df = cause_df.loc[~(cause_df['acause'].isin(no_rd_acauses))]
    cause_list = list(cause_df.cause_id.unique())
    return cause_list


def agg_secret_causes(df, **cache_kwargs):
    """Replace secret causes with parent.

    Currently specific to injuries secret causes, but could be
    rewritten to be more flexible
    """
    df = add_cause_metadata(
        df, ['secret_cause'], cause_set_version_id=CONF.get_id(
            'cause_set_version'), **cache_kwargs
    )
    chh = get_current_cause_hierarchy(
        cause_set_id=4, gbd_round_id=None, **cache_kwargs
    )
    sch = chh.loc[(chh["acause"].str.contains("inj")) & (
        chh["secret_cause"] == 1)]
    secret_id = sch.set_index('cause_id')['parent_id'].to_dict()
    parent_name = sch.set_index('parent_id')['acause_parent'].to_dict()
    df['parent_id'] = df['cause_id'].map(secret_id)
    df['acause_parent'] = df['parent_id'].map(parent_name)
    df.loc[(df["secret_cause"] == 1) & (
        df["cause_id"] != 743), "cause_id"] = df["parent_id"]
    df.loc[(df["secret_cause"] == 1) & (df["cause_id"] != 743),
           "acause"] = df["acause_parent"]
    df.drop(["secret_cause", "parent_id", "acause_parent"],
            axis=1, inplace=True)

    return df


def create_age_standardized_rates(df, draw_cols, dem_cols, location_set=40):
    """Standard function for creating age standardized age groups.

    Arguments
        df - pandas dataframe, must have at least age, sex,
        location, year columns
        draw_cols - the value columns to collapse
        dem_cols - identifying columns to collapse by
        location_set - always add pop for location_set_id 8 (standard), but
        if there are any missing values, then will add population
        for this optional location set
    Returns
        a dataframe with only age_group_id 27
    """
    # set block rerun kwargs
    cache_kwargs = {'force_rerun': False, 'block_rerun': True}

    # return a new dataframe, don't alter existing
    df = df.copy()

    # only keep detailed age groups
    detailed_age_ids = list(get_cod_ages(**cache_kwargs)
                            ['age_group_id'].unique())
    df = df.loc[df['age_group_id'].isin(detailed_age_ids)]

    # pull in the standard population weights
    age_weight_df = get_age_weights(force_rerun=False, block_rerun=True)
    age_weight_dict = age_weight_df.drop_duplicates(
        ['age_group_id', 'age_group_weight_value']
    ).set_index('age_group_id')['age_group_weight_value'].to_dict()
    df['weight'] = df['age_group_id'].map(age_weight_dict)
    report_if_merge_fail(df, 'weight', 'age_group_id')

    # merge on population
    pop_run_id = CONF.get_id('pop_run')
    # default location set is 8
    df = add_population(df, pop_run_id=pop_run_id, **cache_kwargs)
    # if there are still some missing, then add SDI location_ids
    null_pop = df['population'].isnull()
    null_pop_df = df[null_pop].drop('population', axis=1)
    if len(null_pop_df) > 0:
        null_pop_df = add_population(null_pop_df, pop_run_id=pop_run_id,
                                     location_set_id=location_set,
                                     **cache_kwargs)
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


def get_parent_child_cause_dict():
    """Pull the list of aggregated causes from GBD reporting hierarchy."""
    cause_df = get_cause_metadata(
        cause_set_id=3, gbd_round_id=CONF.get_id('gbd_round'))
    cause_df = cause_df.loc[(cause_df['yld_only'] != 1) & (
        cause_df['most_detailed'] == 0)]
    level_id_dict = cause_df.groupby('level')['cause_id'].apply(list).to_dict()
    return level_id_dict
