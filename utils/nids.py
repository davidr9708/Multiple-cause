import pandas as pd
from cod_prep.claude.configurator import Configurator
from cod_prep.utils.cod_db_tools import (
    get_function_results,
    add_tbl_metadata
)
from cod_prep.utils.misc import report_if_merge_fail
from cod_prep.downloaders.locations import add_location_metadata
from db_tools import ezfuncs
CONF = Configurator()


def get_datasets(nid=None, extract_type_id=None, source=None,
                 nid_extract_records=None, location_id=None, year_id=None,
                 data_type_id=None, code_system_id=None, iso3=None, location_set_id=None,
                 location_set_version_id=None, region_id=None, parent_nid=None, is_active=None,
                 verbose=False, force_rerun=False, block_rerun=True):
    """Get nid metadata for datasets according to given filters.

    Special Arguments (selected ones that aren't intuitive):
        nid_extract_records, list of 2-element nid, extract_type_id tuples
            This allows you to get the information in get_datasets for an
            already specified set of nid-extracts. All the other arguments
            that filter data will still be used, so you can pass a list of
            nid extracts, and then filter to only the ones you passed that are
            from year_id=2010, for example.

    Throws:
        AssertionError if no datasets match your specification.
    """
    if is_active is not None:
        assert isinstance(is_active, bool), \
            "Pass either True or False to is_active"
        is_active = 1 * is_active

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
        'parent_nid': parent_nid,
        'is_active': is_active
    }
    # remove the None values
    dataset_filters = {
        k: dataset_filters[k] for k in dataset_filters
        if dataset_filters[k] is not None and k is not 'nid_extract_records'
    }
    use_nid_extract_records = False
    if nid_extract_records is not None:
        use_nid_extract_records = True

    cache_options = {
        'verbose': verbose,
        'force_rerun': force_rerun,
        'block_rerun': block_rerun,
        'cache_results': False,
        'cache_dir': 'standard'
    }

    datasets = get_nidlocyear_map(**cache_options)
    add_cols = ['source', 'data_type_id', 'code_system_id', 'parent_nid', 'is_active']
    datasets = add_nid_metadata(datasets, add_cols, **cache_options)
    for need_col in ['source', 'data_type_id', 'code_system_id', 'is_active']:
        report_if_merge_fail(datasets, need_col, ['nid', 'extract_type_id'])

    # always need a location set to attempt addition of ihme_loc_id
    if location_set_version_id is None:
        location_set_version_id = CONF.get_id('location_set_version')
    datasets = add_location_metadata(
        datasets, ['ihme_loc_id', 'region_id'],
        location_set_version_id=location_set_version_id,
        **cache_options
    )

    if location_set_id is not None:
        # filter on just the datasets that matched the relevant hierarchy
        datasets = datasets.loc[datasets['ihme_loc_id'].notnull()]
    datasets['iso3'] = datasets['ihme_loc_id'].str.slice(0, 3)

    if use_nid_extract_records:
        nid_extract_df = pd.DataFrame.from_records(
            nid_extract_records, columns=['nid', 'extract_type_id']
        )
        datasets = datasets.merge(nid_extract_df)

    for var in list(dataset_filters.keys()):
        vals = dataset_filters[var]
        if not isinstance(vals, list):
            vals = [vals]
        datasets = datasets.loc[(datasets[var].isin(vals))]

    if len(datasets) == 0:
        raise AssertionError(
            "Given dataset filters produced no "
            "datasets: \n{}".format(dataset_filters)
        )

    return datasets


def get_nidlocyear_map(**cache_kwargs):
    """Pull a lookup from nid to the locations and years contained.

    Optional caching utility included.

    !Attention!: as with all queries to the database, do not use this
        function with cached=False in parallel.

    Arguments:
        force_rerun, bool: whether to force the method to rerun the query
        block_rerun, bool: whether the force reading from a cached file
            (if both of above are false, the file will be tried and if not
            found the database will be queried; if both are true an error will
            be thrown)
        cache_results, bool: whether to save results to a cached file
        cache_dir: str: a directory for searching or saving cached results

    Throws:
        IOError if search for cached version fails

    Returns:
        df, pandas DataFrame: nid, location_id, year_id
    """
    nid_query = """SELECT nid, extract_type_id, location_id, year_id, representative_id
                    FROM cod.mcause_nid_location_year"""
    function = ezfuncs.query
    args = [nid_query]
    kwargs = {'conn_def': CONF.get_database_setup('db')}
    cache_name = "nid_locyears"

    # return query results based on cache option
    df = get_function_results(
        function,
        args,
        kwargs,
        cache_name,
        **cache_kwargs
    )

    return df


def get_nid_metadata(**cache_kwargs):
    """Fetch claude metadata for nid.

    Optional caching utility included.

    !Attention!: as with all large queries to the database, do not use this
        function with cached=False in parallel.

    Arguments:
        force_rerun, bool: whether to force the method to rerun the query
        block_rerun, bool: whether the force reading from a cached file
            (if both of above are false, the file will be tried and if not
            found the database will be queried; if both are true an error will
            be thrown)
        cache_results, bool: whether to save results to a cached file
        cache_dir: str: a directory for searching or saving cached results

    Throws:
        IOError if search for cached version fails

    Returns:
        df, pandas DataFrame: the metadata associated with each nid:
            ghdx_title, the name of the nid entry as appears in GHDx
            ghdx_coverage, the coverage (e.g. "Subnational", "National") as it
                appears in the GHDx
    """
    nid_query = """
        SELECT * FROM cod.mcause_nid_metadata
    """
    function = ezfuncs.query
    args = [nid_query]
    kwargs = {'conn_def': CONF.get_database_setup('db')}
    cache_name = "nid_metadata"

    # return query results based on cache option
    df = get_function_results(
        function,
        args,
        kwargs,
        cache_name,
        **cache_kwargs
    )
    return df


def add_nid_metadata(df, add_cols, merge_col=['nid', 'extract_type_id'],
                     nid_meta_df=None, **cache_kwargs):
    """Add ghdx info from nid.

    Some of the core columns available to add:
        code_system_id
        data_type_id
        parent_nid
        representative_id
        source

    No merge checking is performed

    Arguments:
        df: dataframe, the dataframe to add metadata to
        add_cols: list or str, the column or columns that you want added
        merge_col: str, the id column that uniquely identifies the age group
            table and is contained in df (only age_group_id works right now)
        **kwargs: only the keyword arguments that are accepted by
            get_nid_metadata

    Returns:
        df: dataframe, the same data with addition of add_cols.

    Throws:
        AssertionError if add_cols are already present or merge_col does not
            uniquely identify the metadata table
    """
    if nid_meta_df is not None:
        tbl_or_function = nid_meta_df
    else:
        tbl_or_function = get_nid_metadata
    return add_tbl_metadata(
        tbl_or_function, df, add_cols, merge_col, **cache_kwargs
    )
