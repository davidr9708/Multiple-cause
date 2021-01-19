"""Create cause aggregates."""

from builtins import str
from builtins import range
import os
import sys
import pandas as pd
import argparse
from cod_prep.utils import print_log_message, report_duplicates
from cod_prep.downloaders import get_current_cause_hierarchy
from cod_prep.claude.configurator import Configurator
from mcod_prep.utils.causes import get_most_detailed_inj_causes
from mcod_prep.age_loc_aggregation import create_age_standardized_rates

CONF = Configurator('standard')
DEM_COLS = ['cause_id', 'location_id', 'sex_id', 'year_id', 'age_group_id']
DRAW_COLS = ["draw_" + str(x) for x in range(0, 1000)]
CACHE_KWARGS = {'force_rerun': False, 'block_rerun': True}


def get_child_causes(cause_meta_df, parent_cause_id):
    child_cause_list = list(cause_meta_df.query(
        'parent_id == {}'.format(parent_cause_id)
    )['cause_id'].unique())
    if parent_cause_id in child_cause_list:
        child_cause_list.remove(parent_cause_id)
    return child_cause_list


def get_cause_list_burden(parent_cause_id, custom):
    """Aggregate up the given cause hierarchy."""
    print_log_message("Pulling cause hierarchy")
    if not custom:
        cause_meta_df = get_current_cause_hierarchy(
            cause_set_version_id=CONF.get_id('reporting_cause_set_version'), **CACHE_KWARGS
        ).query('yld_only != 1')
    else:
        raise NotImplementedError
        # !!! TO DO !!! build this out!!!
        # cause_meta_df = get_sepsis_cause_hierarchy(
        #     custom=True, **cache_kwargs
        # ).query("most_detailed == 1")[['level_1', 'level_2']]
        # cause_meta_df.rename(
        #     columns={'level_1': 'parent_id', 'level_2': 'cause_id'}, inplace=True)

    # get a list of all child causes given the parent_cause_id
    cause_list = get_child_causes(cause_meta_df, parent_cause_id)

    return cause_list


def get_cause_list_rd_props(int_cause, parent_dir):
    """Pull in the list of possible targets."""
    # set of causes that are never allowed to be targets, per Mohsen
    # some, we adjust elsewhere in the pipeline, others are estimated by shocks, etc.
    bad_targets = [500, 543, 544, 729, 945]
    if int_cause in ['x59', 'y34']:
        cause_list = get_most_detailed_inj_causes(
            CACHE_KWARGS, int_cause,
            cause_set_version_id=CONF.get_id('reporting_cause_set_version'))
    else:
        cause_list = list(
            pd.read_csv(os.path.join(parent_dir, "cause_list.csv")).keep_causes.unique()
        )
        # Mohsen wants to remove HIV as a target for heart failure
        if '_hf' in int_cause:
            bad_targets += [298]
        # remove sids as a target for acute respiratory failure
        elif int_cause == 'arf':
            bad_targets += [686]
        # remove neo_other_benign as target for unspecified CNS + pneumonitis
        elif int_cause in ['unsp_cns', 'pneumonitis']:
            bad_targets += [490]
            # drop suicide by firearm, cvd_pvd as target for penumonitis
            if int_cause == 'pneumonitis':
                bad_targets += [721, 502]
    cause_list = list(set(cause_list) - set(bad_targets))
    return cause_list


def main(out_dir, year_id, int_cause, product, parent_cause_id, custom):
    print(locals())
    if product == 'rd_props':
        parent_dir = out_dir.rsplit('/', 1)[0]
        cause_list = get_cause_list_rd_props(int_cause, parent_dir)
        agg_cause = '_all_int_cause'
    elif product == 'burden':
        cause_list = get_cause_list_burden(parent_cause_id, custom)
        agg_cause = parent_cause_id
    else:
        raise NotImplemented

    print_log_message("Reading in age/sex/location aggregated deaths files")
    cause_dfs = []
    for cause_id in cause_list:
        df = pd.read_csv(f"{out_dir}/{cause_id}_aggregate.csv")
        # drop age standardized rows, we'll recreate these
        df = df.query('age_group_id != 27')
        cause_dfs.append(df)
    df = pd.concat(cause_dfs, ignore_index=True, sort=True)
    df['cause_id'] = agg_cause
    df = df.groupby(DEM_COLS, as_index=False)[DRAW_COLS].sum()

    print_log_message("Creating age standardized rates")
    age_std_df = create_age_standardized_rates(df, DRAW_COLS, DEM_COLS)
    df = pd.concat([age_std_df, df], ignore_index=True, sort=True)

    report_duplicates(df, DEM_COLS)
    df[DEM_COLS] = df[DEM_COLS].astype(int)

    print_log_message("Saving output")
    df.to_csv(f"{out_dir}/{agg_cause}_aggregate.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate causes to parent')
    parser.add_argument('out_dir', type=str)
    parser.add_argument('year_id', type=int)
    parser.add_argument('int_cause', type=str)
    parser.add_argument('product', type=str)
    # only need these arguments if the product == "burden"
    parser.add_argument('--parent_cause_id', type=int)
    parser.add_argument('--custom', nargs=1, type=bool, default=False)
    args = parser.parse_args()
    main(**vars(args))
