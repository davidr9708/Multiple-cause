"""Read in split groups and pass them through redistribution.

Split groups are created from run_pipeline_redistribution_master.py
and passed to this script, which then passes them through the
"guts" of redistribution. The master script then appends each
split group togehter post redistribution.
"""
import sys
import pandas as pd
import time
from mcod_prep.utils.nids import get_datasets
from cod_prep.claude.redistribution import GarbageRedistributor
from cod_prep.claude.configurator import Configurator
from cod_prep.downloaders.nids import get_value_from_nid
from cod_prep.utils import print_log_message

CONF = Configurator('standard')
PACKAGE_DIR = CONF.get_directory('rd_package_dir').format(refresh=CONF.get_id('refresh'))
SG_DIR = CONF.get_directory('rd_process_data') + '/{nid}/{extract_type_id}/{int_cause}/split_{sg}'


def read_cause_map(code_system_id):
    """Read in cause map csv produced by downloading packages."""
    df = pd.read_csv(f'{PACKAGE_DIR}/{code_system_id}/cause_map.csv')
    return df


def read_split_group(nid, extract_type_id, sg, int_cause):
    """Read in split group dataframe."""
    indir = SG_DIR.format(nid=nid, extract_type_id=extract_type_id, int_cause=int_cause, sg=sg)
    df = pd.read_csv('{}/for_rd.csv'.format(indir), dtype={'cause': 'object'})
    return df


def write_split_group(df, nid, extract_type_id, sg, int_cause):
    """Write completed split group."""
    indir = SG_DIR.format(nid=nid, extract_type_id=extract_type_id, int_cause=int_cause, sg=sg)
    df.to_csv('{}/post_rd.csv'.format(indir), index=False)


def run_pipeline(df, nid, extract_type_id, cause_map, code_system_id, sg,
                 data_type_id, int_cause, write_diagnostics=True):
    """Run full pipeline, chaining together redistribution processes."""
    signature_ids = [
        'global', 'dev_status', 'super_region', 'region', 'country', 'subnational_level1',
        'subnational_level2', 'location_id', 'site_id', 'year_id', 'sex', 'age', 'age_group_id',
        'nid', 'extract_type_id', 'split_group', 'sex_id', int_cause
    ]
    proportion_ids = [
        'global', 'dev_status', 'super_region', 'region', 'country', 'subnational_level1',
        'site_id', 'year_id', 'sex', 'age', 'nid', 'extract_type_id', 'split_group', int_cause
    ]
    output_cols = [
        'location_id', 'site_id', 'year_id', 'nid', 'extract_type_id', 'split_group', 'cause',
        'freq', 'sex_id', 'age_group_id', int_cause
    ]

    if 'died' in df.columns:
        for id_list in [signature_ids, proportion_ids, output_cols]:
            id_list.append('died')

    redistributor = GarbageRedistributor(
        code_system_id, package_dir=PACKAGE_DIR, signature_ids=signature_ids,
        proportion_ids=proportion_ids, output_cols=output_cols, first_and_last_only=False
    )
    df = redistributor.get_computed_dataframe(df, cause_map)

    if write_diagnostics:
        outdir = SG_DIR.format(nid=nid, extract_type_id=extract_type_id, int_cause=int_cause, sg=sg)

        signature_metadata = redistributor.get_signature_metadata()
        signature_metadata.to_csv(
            "{}/signature_metadata.csv".format(outdir), index=False
        )

        proportion_metadata = redistributor.get_proportion_metadata()
        proportion_metadata.to_csv(
            "{}/proportion_metadata.csv".format(outdir), index=False
        )

        magic_table = redistributor.get_diagnostic_dataframe()
        magic_table.to_csv(
            "{}/magic_table.csv".format(outdir), index=False
        )

    return df


def main(nid, extract_type_id, split_group, code_system_id, int_cause):
    """Main method."""
    start_time = time.time()
    df = read_split_group(nid, extract_type_id, split_group, int_cause)
    cause_map = read_cause_map(code_system_id)
    data_type_id = get_value_from_nid(
        nid, 'data_type_id', nid_meta_df=get_datasets(nid, extract_type_id)
    )
    df = run_pipeline(df, nid, extract_type_id, cause_map,
                      code_system_id, split_group, data_type_id, int_cause)
    write_split_group(df, nid, extract_type_id, split_group, int_cause)
    run_time = time.time() - start_time
    print_log_message(f"Total run time: {run_time}")


if __name__ == "__main__":
    nid = int(sys.argv[1])
    extract_type_id = int(sys.argv[2])
    split_group = int(sys.argv[3])
    code_system_id = int(sys.argv[4])
    int_cause = str(sys.argv[5])
    main(nid, extract_type_id, split_group, code_system_id, int_cause)
