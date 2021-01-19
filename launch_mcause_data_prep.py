"""Launch formatting, mapping, and redistribution steps for multilple cause data."""

from __future__ import print_function
import argparse
from builtins import object
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from mcod_prep.utils.nids import get_datasets
from mcod_prep.mcod_mapping import MCoDMapper
from cod_prep.downloaders import (
    get_map_version, get_cause_map, get_current_cause_hierarchy, get_ages,
    get_current_location_hierarchy, get_remove_decimal
)
from cod_prep.claude.configurator import Configurator
from cod_prep.utils import report_duplicates
from cod_prep.claude.claude_io import delete_claude_output


class MCauseLauncher(object):

    conf = Configurator('standard')
    cache_options = {
        'force_rerun': True,
        'block_rerun': False,
        'cache_dir': "standard",
        'cache_results': True,
        'verbose': True

    }

    source_memory_dict = {
        'MEX_INEGI': '7G', 'BRA_SIM': '12G', 'USA_NVSS': '20G',
        'ITA_ISTAT': '8G', 'COL_DANE': '2G'
    }

    def __init__(self, run_filters):
        self.run_filters = run_filters
        self.location_set_version_id = self.conf.get_id('location_set_version')
        self.cause_set_version_id = self.conf.get_id('cause_set_version')
        self.mcod_code = self.conf.get_directory('mcod_code')

    def prep_run_filters(self):
        datasets_kwargs = {'force_rerun': True, 'block_rerun': False}
        datasets_kwargs.update(
            {k: v for k, v in self.run_filters.items() if k not in [
                'intermediate_causes', 'phases']}
        )
        datasets = get_datasets(**datasets_kwargs)
        datasets = datasets.drop_duplicates(
            ['nid', 'extract_type_id']).set_index(
            ['nid', 'extract_type_id'])[['year_id', 'code_system_id', 'source', 'data_type_id']]
        datasets['code_map_version_id'] = datasets['code_system_id'].apply(
            lambda x: get_map_version(x, 'YLL', 'best')
        )
        datasets['remove_decimal'] = datasets['code_system_id'].apply(
            lambda x: get_remove_decimal(x)
        )
        return datasets

    def cache_resources(self, cache_functions_to_run_with_args):
        """Cache metadata files."""
        for cache_function, kwargs in cache_functions_to_run_with_args:
            function_name = cache_function.__name__
            cache_exists = cache_function(only_check_cache_exists=True, verbose=True, **kwargs)
            if cache_exists:
                print(f"No need to recache method {function_name} with args: {kwargs}")
            else:
                print(f"Running {function_name} with args: {kwargs}")
                kwargs.update(self.cache_options)
                cache_function(**kwargs)

    def launch_format_map(self, year, source, int_cause, code_system_id,
                          code_map_version_id, nid, extract_type_id, data_type_id):
        """Submit qsub for format_map phase."""
        delete_claude_output('format_map', nid, extract_type_id, sub_dirs=int_cause)
        worker = f"{self.mcod_code}/run_phase_format_map.py"
        params = [int(year), source, int_cause, int(code_system_id), int(code_map_version_id),
                  int(self.cause_set_version_id), int(nid), int(extract_type_id), int(data_type_id)]
        jobname = f'mcause_format_map_{source}_{year}_{int_cause}'
        try:
            memory = self.source_memory_dict[source]
        except KeyError:
            print(f"{source} is not in source_memory_dict. Trying with 5G.")
            memory = '5G'

        if data_type_id == 3:
            runtime = '02:00:00'
        else:
            runtime = '00:30:00'

        jid = submit_mcod(
            jobname, 'python', worker, cores=1, memory=memory, params=params,
            verbose=True, logging=True, jdrive=True, runtime=runtime
        )
        return jid

    def launch_redistribution(self, nid, extract_type_id, code_system_id, code_map_version_id,
                              remove_decimal, data_type_id, int_cause, holds=[]):
        """Submit qsub for redistribution."""
        delete_claude_output('redistribution', nid, extract_type_id, sub_dirs=int_cause)
        worker = f"{self.mcod_code}/run_phase_redistribution.py"
        jobname = f"mcause_redistribution_{nid}_{extract_type_id}"
        params = [nid, extract_type_id, self.cause_set_version_id,
                  self.location_set_version_id, code_system_id, code_map_version_id,
                  remove_decimal, int(data_type_id), int_cause]
        submit_mcod(jobname, 'python', worker, cores=1, memory='2G', params=params,
                    holds=holds, verbose=True, logging=True)

    def launch(self):
        datasets = self.prep_run_filters()
        cache_functions_to_run_with_args = [
            (get_current_cause_hierarchy, {'cause_set_version_id': self.cause_set_version_id}),
            (get_ages, {}),
            (get_current_location_hierarchy,
                {'location_set_version_id': self.location_set_version_id})
        ]
        for code_map_version_id in list(datasets.code_map_version_id.unique()):
            cache_functions_to_run_with_args.append(
                (get_cause_map, {'code_map_version_id': code_map_version_id})
            )
        self.cache_resources(cache_functions_to_run_with_args)

        # run things!
        format_map_jobs = []
        for row in datasets.itertuples():
            nid, extract_type_id = row.Index
            for int_cause in self.run_filters['intermediate_causes']:
                if 'format_map' in self.run_filters['phases']:
                    jid = self.launch_format_map(
                        row.year_id, row.source, int_cause, row.code_system_id,
                        row.code_map_version_id, nid, extract_type_id, row.data_type_id
                    )
                    format_map_jobs.append(jid)
                if 'redistribution' in self.run_filters['phases']:
                    self.launch_redistribution(
                        nid, extract_type_id, row.code_system_id, row.code_map_version_id,
                        row.remove_decimal, row.data_type_id, int_cause, holds=format_map_jobs
                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NID/etids for which to run formatting, mapping, and redistribution')
    parser.add_argument('--intermediate_causes', help='intermediate cause(s) of interest',
                        required=True, nargs='+', choices=MCoDMapper.possible_int_causes)
    parser.add_argument('--phases', help='data processing phases', required=True,
                        nargs='+', choices=['format_map', 'redistribution'])
    parser.add_argument('--iso3', nargs='*')
    parser.add_argument('--code_system_id', nargs='*')
    parser.add_argument('--data_type_id', nargs='*')
    parser.add_argument('--year_id', type=int, nargs='*')
    parser.add_argument('--source', nargs='*')
    parser.add_argument('--nid', nargs='*', type=int)
    args = parser.parse_args()
    launcher = MCauseLauncher(vars(args))
    launcher.launch()
