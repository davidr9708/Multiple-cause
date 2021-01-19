from __future__ import print_function
from builtins import zip
import os
import pandas as pd
import numpy as np
import re
from cod_prep.utils import (
    print_log_message, report_duplicates, clean_icd_codes, report_if_merge_fail, warn_if_merge_fail
)
from cod_prep.downloaders import get_cause_map, add_code_metadata
from cod_prep.claude.configurator import Configurator
from mcod_prep.utils.nids import get_datasets


class MCoDMapper():
    """Map ICD codes to code_ids, cause_ids.

    Arguments:
        int_cause (str): the intermediate cause of interest (e.g. sepsis)
        code_system_id (int): the ICD category, determines which map to use
        code_map_version_id (int): the version of the map to use
        df (dataframe): dataframe of formatted mcod data

    Returns:
        df (dataframe): dataframe with the underlying cause mapped to code id and cause_id
        and the causes in the chain flagged for containing the intermediate cause of interest.


    """
    cache_options = {'force_rerun': False, 'block_rerun': True}
    conf = Configurator()
    inj_causes = ['x59', 'y34']
    int_cause_name_dict = {
        'pulmonary_embolism': ['pulmonary embolism'], 'right_hf': ['right heart failure'],
        'left_hf': ['left heart failure'], 'unsp_hf': ['unspecified heart failure'],
        'arterial_embolism': ['arterial embolism'], 'aki': ['acute kidney failure'],
        'hf': ['right heart failure', 'left heart failure', 'unspecified heart failure'],
        'embolism': ['arterial embolism', 'pulmonary embolism'],
        'x59': ['unspecified external factor x59'],
        'y34': ['external causes udi,type unspecified-y34'], 'drug_overdose': ['drug_overdose'],
        'sepsis': ['sepsis'], 'hepatic_failure': ['hepatic failure'],
        'arf': ['acute respiratory failure'], 'pneumonitis': ['pneumonitis'],
        'unsp_cns': ['unspecified cns signs and symptom'], 'infectious_syndrome': ''
    }
    possible_int_causes = list(int_cause_name_dict.keys())

    def __init__(self, int_cause, code_system_id, code_map_version_id, drop_p2):
        self.int_cause = int_cause
        self.code_system_id = code_system_id
        self.code_map_version_id = code_map_version_id
        self.drop_p2 = drop_p2
        assert self.int_cause in self.possible_int_causes, \
            f"{self.int_cause} is not a valid intermediate cause"
        self.full_cause_name = self.int_cause_name_dict[self.int_cause]
        if type(self.full_cause_name) != list:
            self.full_cause_name = [self.full_cause_name]

    @staticmethod
    def get_code_columns(df):
        """Get a list of raw cause columns with ICD codes as values."""
        col_names = list(df.columns)
        code_cols = [x for x in col_names if "multiple_cause" in x and "pII" not in x] + ['cause']
        return code_cols

    @staticmethod
    def _get_cause_num(mcod_col):
        """Get sort order for cause columns.

        Assumes you have an underlying cause (cause_x) column and chain columns (multiple_cause_x)
        and that the value to sort off of is after the second underscore.
        """
        if mcod_col.startswith('cause'):
            return '0'
        else:
            assert re.match(r"^multiple_cause_[a-z]*[0-9]*", mcod_col), \
                f"column {mcod_col} does not match expected format: multiple_cause_x"
            return mcod_col.split('_')[2]

    @staticmethod
    def prep_raw_mapped_cause_dictionary(raw_cols, mapped_cols):
        """Create dictionary of raw cause columns to mapped cause columns.

        Ensures that "multiple_cause_2_mapped" is the value associated with
        "multiple_cause_2" key, e.g.
        """
        raw_cols = sorted(raw_cols, key=MCoDMapper._get_cause_num)
        mapped_cols = sorted(mapped_cols, key=MCoDMapper._get_cause_num)
        return dict(list(zip(raw_cols, mapped_cols)))

    @staticmethod
    def fix_icd_codes(df, codes, code_system_id):
        """Adjustment to icd9/10 cause codes."""
        if code_system_id == 6:
            # codes between 800 to 900 need an E if underlying
            # assume 800, 900 codes are N codes if in the chain, don't add any prefix
            df.loc[df['cause'].str.contains('^[89]'), 'cause'] = 'E' + df['cause']
        elif code_system_id == 1:
            # S + T codes are always intermediate causes of death
            # V + Y codes are always the underlying cause of death
            violations = df['cause'].str.contains('^[ST]')
            num_violations = len(df[violations])
            if num_violations > 0:
                print_log_message(
                    f"Found S or T code as underlying cause, dropping {num_violations} rows"
                )
                # ensure this is a small proportion of the data
                assert np.isclose(len(df[~violations]), len(df), rtol=.11)
                df = df.loc[~violations]

            # next check violations in chain causes
            # V and Y codes can only be UCOD
            for col in codes:
                if col != 'cause':
                    violations = df[col].str.contains('^[VY]')
                    num_violations = len(df[violations])
                    if num_violations > 0:
                        print_log_message(
                            f"Setting {num_violations} rows with V/Y in chain to 0000 for {col}")
                        df.loc[violations, col] = '0000'
        return df

    @staticmethod
    def prep_cause_package_map(cause_package_map):
        """Expects cause-package map.

        Set dictionary of value: map_id since we only care about the package name
        or the cause_id, not the individual ICD code level code.
        """
        check_map = cause_package_map[['map_id', 'map_type']].drop_duplicates()
        report_duplicates(check_map, 'map_id')
        cause_package_map = cause_package_map.set_index('value')['map_id'].to_dict()
        return cause_package_map

    @staticmethod
    def prep_cause_map(cause_map):
        """Clean up cause map."""
        cause_map['value'] = clean_icd_codes(cause_map['value'], remove_decimal=True)
        # duplicates are a result of weird _gc, the duplicates dropped all
        # have the higher sort_order (999999)
        cause_map = cause_map.drop_duplicates(['code_system_id', 'value'])
        cause_map['code_id'] = cause_map['code_id'].astype(int)
        cause_map = cause_map.set_index('value')['code_id'].to_dict()
        return cause_map

    @staticmethod
    def map_cause_codes(df, coi_map, coi, cols_to_map=None):
        """Map cause codes to any given value (e.g. acause, category, etc.).

        Inputs
        df (pd dataframe): incoming, unmapped data with ICD codes
        cause_map (pd dataframe): primary cause map, probably downloaded from the engine room
        coi_map (pd dataframe): special map designed just for one cause of interest
        coi (string): cause of interest
        Returns
        df (pd dataframe): mapped dataframe with additional columns for each cause
        """
        df = df.copy()
        if not cols_to_map:
            cols_to_map = MCoDMapper.get_code_columns(df)
        # map chain causes using cause of interest map
        for col in cols_to_map:
            df[col] = df[col].fillna('0000')
            df[col] = df[col].astype(object)
            df[col + '_' + coi] = df[col].map(coi_map)
        return df

    @staticmethod
    def trim_and_remap(df, code_dict, cause_map, code_system_id):
        """Trim ICD codes to 4 digits, map again, then 3, and map again."""
        df = df.copy()
        # before trimming, map "null" chain causes to '0000'
        for code, mapped_code in list(code_dict.items()):
            df.loc[df[code] == '0000', mapped_code] = '0000'

        # trim and re map null mappings
        for n in reversed(range(3, 6)):
            for code, mapped_code in list(code_dict.items()):
                try:
                    df.loc[df[mapped_code].isnull(), code] = df[code].apply(lambda x: x[0:n])
                except TypeError:
                    # was getting a type error for some unicode issues?
                    if mapped_code != 'cause_mapped':
                        df[mapped_code] = '0000'
                    else:
                        print("problem code here..." + df[code])
                df.loc[df[mapped_code].isnull(), mapped_code] = df[code].map(cause_map)
        return df

    def prep_int_cause_map(self):
        map_dir = self.conf.get_directory('process_inputs')
        code_system_name = {1: 'icd10', 6: 'icd9'}[self.code_system_id]
        if self.int_cause == 'sepsis':
            df = pd.read_excel(f"{map_dir}/sepsis_map_{code_system_name}.xlsx",
                               sheet_name=1, dtype={'icd_code': object})

            df['icd_code'] = clean_icd_codes(df['icd_code'], remove_decimal=True)
            df = df[['icd_code', 'total_sepsis']].drop_duplicates()
            df['total_sepsis'] = df['total_sepsis'].str.lower()
            df['total_sepsis'] = df['total_sepsis'].fillna('no_sepsis')
            df['total_sepsis'] = df['total_sepsis'].astype(str)

            df.loc[
                df['total_sepsis'] == 'sepsis_inf and sepsis_maternal_neonatal',
                'total_sepsis'] = 'sepsis_infectious'

            mcod_map = dict(list(zip(df['icd_code'], df['total_sepsis'])))
            mcod_map.update({'acause_inj_trans_road_4wheel': 'no_sepsis'})

        elif self.int_cause == 'infectious_syndrome':
            mcod_map = self.prep_infectious_syndrome_map()

        else:
            df = pd.read_excel(f"{map_dir}/{mcause_map}.xlsx", dtype={'icd_code': object})
            df = df[['icd_code', 'package_description', 'code_system']].drop_duplicates()

            # cleanup strings and things
            df['icd_code'] = clean_icd_codes(df['icd_code'], remove_decimal=True)
            df[['package_description', 'code_system']] = \
                df[['package_description', 'code_system']].str.lower()
            df['package_description'] = df['package_description'].astype(str)

            # only keep the rows we need for this intermediate cause
            df = df.loc[df['package_description'].isin(self.full_cause_name)]

            # intermediate causes should be mutually exclusive
            report_duplicates(df, ['icd_code', 'code_system'])

            # subset to just the code system being run through
            df = df.query(f'code_system == "{code_system_name}"')

            assert len(df) > 0, \
                f"There are no mappings for {code_system_name}, {self.full_cause_name}"

            # convert to a dictionary
            mcod_map = dict(list(zip(df['icd_code'], df['package_description'])))

        return mcod_map

    def capture_int_cause(self, df, int_cause_cols):
        """Flag deaths related to the intermediate cause."""
        df[self.int_cause] = None

        if self.int_cause == 'sepsis':
            ucod_infectious = (
                (df['cause_sepsis'].str.contains('inf')) |
                (df['cause_sepsis'].str.contains('maternal_neonatal'))
            )

            for col in int_cause_cols:
                # for when not all chain causes successfully map
                df[col] = df[col].fillna("no_sepsis")

                # implict sepsis cases when ucod is inf and any column is organ disfunction
                df.loc[(df[col].str.contains('odf')) & ucod_infectious, 'sepsis'] = 'implicit'

                # explicit sepsis takes priority, so set these last
                df.loc[df[col].str.contains('maternal_neonatal'), 'sepsis'] = 'explicit'
                df.loc[df[col] == "sepsis_explicit", "sepsis"] = "explicit"

            df['sepsis'] = df['sepsis'].fillna('no_sepsis')

        else:
            for col in int_cause_cols:
                df[col] = df[col].fillna("other")
                df.loc[df[col].isin(self.full_cause_name), self.int_cause] = 1
            df[self.int_cause] = df[self.int_cause].fillna(0)

        assert df[self.int_cause].notnull().values.all()

        return df

    def _remove_none(self, lst):
        """Helper function to remove 'none' values in a list."""
        if len(lst) > 1:
            try:
                lst.remove('none')
            except ValueError:
                return lst
        return lst

    def set_part2_flag(self, df):
        """Mark whether or not the cause of interest is from part 2 of the death certificate."""
        p2_cols = [x for x in df.columns if 'pII' in x]
        int_cause_chains = [x for x in df.columns if (self.int_cause in x) and ('multiple' in x)]
        p2_chain_dict = dict(list(zip(p2_cols, int_cause_chains)))
        df['pII_' + self.int_cause] = 0
        for p2_col, chain in sorted(p2_chain_dict.items()):
            df.loc[
                (df[chain].isin(self.full_cause_name)) &
                (df[p2_col] == 1), 'pII_' + self.int_cause
            ] = 1
        return df

    def get_computed_dataframe(self, df, map_underlying_cause=True):
        """Return mapped dataframe."""
        # list of all cause columns
        raw_cause_cols = MCoDMapper.get_code_columns(df)
        df = MCoDMapper.fix_icd_codes(df, raw_cause_cols, self.code_system_id)

        if map_underlying_cause:
            print_log_message("Mapping underlying cause/primary diagnosis")
            cause_map = get_cause_map(
                code_map_version_id=self.code_map_version_id, **self.cache_options
            )
            code_map = MCoDMapper.prep_cause_map(cause_map)
            df['cause_mapped'] = df['cause'].map(code_map)

            print_log_message("Trimming ICD codes and remapping underlying cause/primary diagnosis")
            df = MCoDMapper.trim_and_remap(
                df, {'cause': 'cause_mapped'}, code_map, self.code_system_id)
            report_if_merge_fail(df, 'cause_mapped', 'cause')

            # merge on the cause_id for the underlying cause
            df = df.rename(columns={'cause_mapped': 'code_id'})
            df['code_id'] = df['code_id'].astype(int)
            df = add_code_metadata(df, 'cause_id', code_map_version_id=self.code_map_version_id,
                                   **self.cache_options)
            report_if_merge_fail(df, 'cause_id', 'code_id')

        print_log_message("Mapping chain causes")
        # get the special intermediate cause map
        int_cause_map = self.prep_int_cause_map()
        df = MCoDMapper.map_cause_codes(df, int_cause_map, self.int_cause)

        print_log_message("Trimming ICD codes and remapping chain causes")
        int_cause_cols = [x for x in df.columns if self.int_cause in x]
        int_cause_col_dict = MCoDMapper.prep_raw_mapped_cause_dictionary(
            raw_cause_cols, int_cause_cols)
        df = MCoDMapper.trim_and_remap(df, int_cause_col_dict, int_cause_map, self.code_system_id)

        # important check for sepsis to correctly distinguish explicit from implicit
        if self.int_cause == 'sepsis':
            report_if_merge_fail(df, 'cause_' + self.int_cause, 'cause')

        print_log_message("Identifying rows with intermediate cause of interest")
        df = self.capture_int_cause(df, int_cause_cols)
        if not self.drop_p2:
            df = self.set_part2_flag(df)

        return df
