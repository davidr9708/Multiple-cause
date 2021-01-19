"""Apply number of deaths post codcorrect to get the number of intermediate cause related deaths."""

from builtins import str
from builtins import range
import sys
import pandas as pd
from get_draws.api import get_draws
from cod_prep.claude.configurator import Configurator
from cod_prep.downloaders import get_current_location_hierarchy, create_age_bins
from cod_prep.utils import print_log_message, report_if_merge_fail

DEM_COLS = ['cause_id', 'location_id', 'sex_id', 'year_id', 'age_group_id']
DRAW_COLS = ["draw_" + str(x) for x in range(0, 1000)]
CONF = Configurator('standard')


def convert_int_cols(df):
    for col in DEM_COLS:
        df[col] = df[col].astype(int)
    return df


def get_location_list():
    lhh = get_current_location_hierarchy(
        location_set_version_id=CONF.get_id('location_set_version'),
        force_rerun=False, block_rerun=True
    ).query("most_detailed == 1")
    return list(lhh['location_id'].unique())


def get_codcorrect_draws(cause_id, year_id, agg_ages):
    """Get draws of deaths from codcorrect by demographic."""
    print_log_message("Pulling draws of codcorrect deaths")
    draws_df = get_draws(
        gbd_id_type='cause_id', gbd_id=cause_id, source='codcorrect', metric_id=1, measure_id=1,
        location_id=get_location_list(), year_id=year_id, gbd_round_id=CONF.get_id('gbd_round'),
        version_id=CONF.get_id('codcorrect'), num_workers=2
    )
    print_log_message("Aggregating draws of deaths to match data")
    # set dropna to True here-- it's OK if the incoming aggregate ages do not cover
    # all existing codcorrect age groups, b/c some causes have restrictions
    # see redistribution_logic_overrides for more information
    draws_df = create_age_bins(draws_df, agg_ages, dropna=True)
    draws_df = draws_df.groupby(DEM_COLS, as_index=False)[DRAW_COLS].sum()
    assert draws_df.notnull().values.all(), "error pulling codcorrect results"
    return draws_df


def calculate_deaths(int_cause_df, deaths_df):
    print_log_message("Calculating intermediate cause deaths")
    # left merge is important-- we have stricter restrictions than codcorrect
    df = int_cause_df.merge(deaths_df, how='left', on=DEM_COLS)
    report_if_merge_fail(df, 'draw_0_y', DEM_COLS)
    for draw in DRAW_COLS:
        df[draw] = df["{}_x".format(draw)] * df["{}_y".format(draw)]
    # drop draw_x, draw_y cols (keep dems and multiplied draws)
    df = df[DEM_COLS + DRAW_COLS]
    assert df.notnull().values.all(), "error calculating deaths"
    return df


def main(out_dir, cause_id, year_id):
    int_cause_df = pd.read_csv(f"{out_dir}/{cause_id}.csv")
    int_cause_df = convert_int_cols(int_cause_df)
    agg_ages = list(int_cause_df.age_group_id.unique())
    draws_df = get_codcorrect_draws(cause_id, year_id, agg_ages)
    df = calculate_deaths(int_cause_df, draws_df)
    print_log_message("Saving output")
    df.to_csv("{}/{}_deaths.csv".format(out_dir, cause_id), index=False)


if __name__ == '__main__':
    out_dir = str(sys.argv[1])
    cause_id = int(sys.argv[2])
    year_id = int(sys.argv[3])
    main(out_dir, cause_id, year_id)
