"""Launches various steps in the pipeline from modeling -> redistribution proportions.

Separated this pipeline from mcod_launch.py for increased flexibility later on. You can
use the outputs from "format_map" of any MCoD data or hospital data for lots of different kinds
of analysis. This pipeline is specifically for creating redistribution proportions.
"""
from __future__ import print_function

from builtins import str
from builtins import object
import os
import datetime
import pandas as pd
import argparse
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from mcod_prep.utils.causes import get_most_detailed_inj_causes
from mcod_prep.utils.covariates import prep_predictions_template
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import makedirs_safely


class BurdenCalculator(object):
    conf = Configurator('standard')
    process_order = ['run_model', 'predict_fractions', 'calculate_deaths', 'calculate_incidence',
                     'age_loc_aggregation', 'cause_aggregation', 'compile_rd_props']
    full_time_series = list(range(1990, 2018))
    product_list = ['mortality', 'rd_props', 'incidence']
    block_rerun = {'force_rerun': False, 'block_rerun': True}
    valid_intermediate_causes = [
        'pulmonary_embolism', 'right_hf', 'left_hf', 'unsp_hf', 'arterial_embolism', 'aki',
        'sepsis', 'explicit_sepsis', 'x59', 'y34', 'hepatic_failure', 'arf', 'pneumonitis',
        'unsp_cns', 'sepsis_fatal_model'
    ]

    def __init__(self, run_filters):
        self.cause_set_version_id = self.conf.get_id('reporting_cause_set_version')
        self.run_filters = self.validate_run_filters(run_filters)
        self.int_causes = self.run_filters['int_causes']
        self.processes = self.run_filters['processes']
        self.description = self.run_filters['description']
        self.end_product = self.run_filters['end_product'][0]
        self.years = self.run_filters['year_id']
        self.custom = self.run_filters['custom']

    def validate_run_filters(self, run_filters):
        if "run_model" in run_filters['processes']:
            assert set(run_filters['processes']) == set(["run_model"]), \
                "Modeling should be run independently for now."

        # convert years to a list of integers
        if run_filters['year_id'] == ['all']:
            run_filters.update({'year_id': self.full_time_series})
        else:
            assert all(isinstance(x, int) for x in run_filters['year_id'])

        # check process order
        ordered_processes = []
        no_more_processes = False
        for process in self.process_order:
            if process in run_filters['processes']:
                # check that this is part of continuous set
                if no_more_processes:
                    raise AssertionError(
                        "There is a gap in the phases called ({}) - must have "
                        "a continuous sequence".format(run_filters['processes'])
                    )
                ordered_processes.append(process)
            else:
                if len(ordered_processes) > 0:
                    # can't have any more phases now, because it would create
                    # a gap in the phase order
                    no_more_processes = True

        # if running the model, then set the description here with the timestamp
        # for all other phases, you have to pass in a model description that exists
        # or the next steps won't know where to get the correct data
        if 'run_model' in run_filters['processes']:
            print("Appending date to model description")
            if run_filters['description'][0] == '':
                description = '{:%Y_%m_%d}'.format(
                    datetime.datetime.now()
                ) + run_filters['description'][0]
            else:
                description = '{:%Y_%m_%d}'.format(
                    datetime.datetime.now()
                ) + '_' + run_filters['description'][0]
            run_filters.update({'description': description})
        else:
            description = run_filters['description'][0]
            run_filters.update({'description': description})
            for int_cause in run_filters['int_causes']:
                # check that this is a real filepath
                check_path = os.path.join(
                    self.conf.get_directory('process_data'), int_cause,
                    run_filters['end_product'], description
                )
                assert os.path.exists(check_path), f"No existing path: {check_path}"

        if 'sepsis_fatal_model' in run_filters['int_causes']:
            assert run_filters['description'] == '2018_11_14_neldermead_rd', \
                'using this keyword for modeling sepsis mortality is deprecated since GBD 2017'

        return run_filters

    def get_cause_list(self, int_cause, base_out_dir):
        if self.run_filters['cause_id'] is not None:
            return self.run_filters['cause_id']
        if self.end_product == 'rd_props':
            if set(self.int_causes).issubset(['x59', 'y34']):
                for int_cause in self.int_causes:
                    cause_list = get_most_detailed_inj_causes(
                        self.block_rerun, int_cause,
                        cause_set_version_id=self.cause_set_version_id
                    )
            else:
                # target list is unique to intermediate cause + model run
                cause_list = list(
                    pd.read_csv(os.path.join(base_out_dir, "cause_list.csv")).keep_causes.unique()
                )
        elif self.end_product in ['mortality', 'incidence']:
            if self.custom:
                raise NotImplementedError
            else:
                cause_list = list(get_cause_hiearchy_history(
                    cause_set_version_id=self.cause_set_version
                ).query('yld_only != 1 & most_detailed == 1').cause_id.unique())
        return cause_list

    def get_parent_child_cause_dict(self):
        """Return a dictionary of {level: [cause_ids]} for all non-detailed (parent) causes."""
        cause_df = get_cause_hiearchy_history(
            cause_set_version_id=self.cause_set_version
        ).query('yld_only != 1 & most_detailed == 0')
        level_id_dict = cause_df.groupby('level')['cause_id'].apply(list).to_dict()
        return level_id_dict

    def remove_existing_output(self, directory, cause_id, suffix):
        """Safely remove existing cause files."""
        filepath = os.path.join(directory, str(cause_id) + suffix + '.csv')
        if os.path.exists(filepath):
            os.unlink(filepath)

    def launch_compile_rd_props(self, int_cause, holds):
        """Launch one big job to compile redistribution proportions."""
        worker = os.path.join(
            self.conf.get_directory('mcod_code'), 'compile_redistribution_proportions.py'
        )
        jobname = "compile_rd_props_" + int_cause
        params = [self.description, int_cause]
        jid = submit_mcod(jobname, 'python', worker, cores=10, verbose=True, logging=True,
                          jdrive=True, memory='25G', params=params, holds=holds)
        return jid

    def launch_cause_aggregator(self, year, int_cause, base_out_dir, holds):
        """Submit job to aggregate intermediate cause related deaths.

        There are 3 different options for cause aggregation. (1) For creating redistribution
        proportions, we only need a single aggregate for a denominator later ("_all_int_cause").
        For a burden estimnation (incidence and mortality), (2) we need causes aggregated
        to every level of the reporting cause hierarchy and for some intermediate causes
        there is a (3) custom cause hierarchy as well.
        """
        worker = os.path.join(self.conf.get_directory('mcod_code'), 'cause_aggregation.py')
        outdir = f'{base_out_dir}/{year}'
        makedirs_safely(outdir)
        kwarg_dict = {'verbose': True, 'logging': True, 'holds': holds, 'runtime': '00:10:00'}
        if self.end_product == 'rd_props':
            jobname = f"cause_agg_{int_cause}_{year}"
            self.remove_existing_output(outdir, "_all_int_cause", "_aggregate")
            kwarg_dict.update({'params': [outdir, year, int_cause, self.end_product]})
            # most need 15G, others can be knocked down
            int_cause_memory_dict = {
                'arterial_embolism': '5G', 'pulmonary_embolism': '12G', 'hepatic_failure': '5G'
            }
            try:
                memory = int_cause_memory_dict[int_cause]
            except KeyError:
                memory = '15G'
            jid = submit_mcod(jobname, 'python', worker, cores=2, memory=memory, **kwarg_dict)
        elif self.end_product == 'mortality':
            if not self.custom:
                cause_level_dict = self.get_parent_child_cause_dict()
            else:
                cause_level_dict = {0: [1, 2, 3]}
            cause_levels = cause_level_dict.keys()
            holds_dict = dict(zip(cause_levels, [[]] * len(cause_levels)))
            for level, cause_list in sorted(cause_level_dict.items(), reverse=True):
                hold_ids = []
                for cause_id in cause_list:
                    filepath = os.path.join(outdir, "{}_aggregate.csv".format(cause_id))
                    if os.path.exists(filepath):
                        os.unlink(filepath)
                    jobname = f'cause_agg_{int_cause}_{year}_{cause_id}'
                    kwarg_dict.update({'params': [outdir, year, int_cause, self.end_product,
                                                  f'--parent_cause_id {cause_id}',
                                                  f'--custom {self.custom}'],
                                       'holds': holds_dict[level] + holds})
                    memory = '6G'
                    if cause_id in [696, 491, 409, 526, 344]:
                        memory = '10G'
                    elif cause_id in [410]:
                        memory = '15G'
                    jid = submit_mcod(jobname, 'python', worker, 1, memory, **kwarg_dict)
                    hold_ids.append(jid)
                    # if this is the last cause before moving on to the next level,
                    # then add this list of job ids to the 'holds' in submit_cod
                    if (cause_id == cause_list[-1]) & (level != 0):
                        holds_dict.update({level - 1: hold_ids})
        return jid

    def launch_location_aggregator(self, year, cause_id, int_cause, base_out_dir, holds):
        """Submit job to aggregate draws of deaths by location, age, sex."""
        worker = os.path.join(self.conf.get_directory('mcod_code'), 'age_loc_aggregation.py')
        jobname = f'agg_{int_cause}_{year}_{cause_id}'
        outdir = f'{base_out_dir}/{year}'
        makedirs_safely(outdir)
        params = [outdir, cause_id, year, self.end_product]
        self.remove_existing_output(outdir, cause_id, "_aggregate")
        jid = submit_mcod(
            jobname, 'python', worker, cores=1, memory='3G', params=params,
            verbose=True, logging=True, jdrive=True, runtime="00:10:00", holds=holds
        )
        return jid

    def launch_deaths_calculator(self, year, cause_id, int_cause, base_out_dir, holds=[]):
        """Submit jobs to convert intermediate cause fractions to deaths."""
        worker = os.path.join(self.conf.get_directory('mcod_code'), 'apply_gbd_deaths.py')
        jobname = f'deaths_{int_cause}_{year}_{cause_id}'
        outdir = f'{base_out_dir}/{year}'
        makedirs_safely(outdir)
        params = [outdir, cause_id, year]
        self.remove_existing_output(outdir, cause_id, "_deaths")
        jid = submit_mcod(jobname, 'python', worker, cores=2, memory='5G', params=params,
                          verbose=True, logging=True, jdrive=True, holds=holds)
        return jid

    def remove_predictions_template(self, year, cause_id, int_cause, base_out_dir):
        """Delete predictions templates."""
        outdir = f'{base_out_dir}/{year}'
        self.remove_existing_output(outdir, cause_id, "_template")

    def launch_predictions(self, year, cause_id, int_cause, model_input_df, base_out_dir):
        """Submit job to predict fractions."""
        worker = os.path.join(self.conf.get_directory('mcod_code'), "run_predictionsworker.R")
        jobname = f"predict_{int_cause}_{year}_{cause_id}"
        outdir = f'{base_out_dir}/{year}'
        makedirs_safely(outdir)
        self.remove_existing_output(outdir, cause_id, "_template")
        prep_predictions_template(year, cause_id, outdir, int_cause,
                                  model_input_df, self.end_product)
        params = [cause_id, self.description, int_cause, year, self.end_product]
        self.remove_existing_output(outdir, cause_id, '')
        if 'sepsis' in int_cause:
            memory = '7.5G'
        else:
            memory = '4G'
        jid = submit_mcod(
            jobname, 'r', worker, cores=2, jdrive=True, params=params, verbose=True,
            memory=memory, logging=True, runtime="00:10:00"
        )
        return jid

    def launch_model(self, int_cause, base_out_dir):
        jobname = int_cause + "_run_model_" + self.description + '_' + self.end_product
        params = [self.description, int_cause, base_out_dir]
        if self.end_product == 'rdp':
            worker = os.path.join(self.conf.get_directory('mcod_code'), 'run_model_rdp.py')
        else:
            worker = os.path.join(self.conf.get_directory('mcod_code'), 'run_model.py')
            params += [self.end_product]
        submit_mcod(jobname, 'python', worker, 1, '25G', params=params,
                    verbose=True, logging=True, jdrive=True, runtime="00:15:00")

    def launch_incidence_calculator():
        # will move my incidence script to the mcod_prep folder
        raise NotImplementedError

    def launch(self):
        # grab the input model dataframe, need this to predict intermediate cause fractions
        predict_jobs = []
        deaths_jobs = []
        age_loc_agg_jobs = []
        cause_agg_jobs = []
        for int_cause in self.int_causes:
            base_out_dir = os.path.join(
                self.conf.get_directory('process_data'), int_cause,
                self.end_product, self.description
            )
            if 'predict_fractions' in self.processes:
                print("reading in model input")
                model_input_df = pd.read_csv('{base_out_dir}/model_input.csv')
            if 'run_model' in self.processes:
                self.launch_model(int_cause, base_out_dir)
            else:
                cause_list = self.get_cause_list(int_cause, base_out_dir)
                for year_id in self.years:
                    for cause_id in cause_list:
                        if 'predict_fractions' in self.processes:
                            jid = self.launch_predictions(year_id, cause_id,
                                                          int_cause, model_input_df)
                            predict_jobs.append(jid)
                        if 'remove_predictions_template' in self.processes:
                            jid = self.remove_predictions_template(year_id, cause_id, int_cause)
                        if 'calculate_deaths' in self.processes:
                            jid = self.launch_deaths_calculator(
                                year_id, cause_id, int_cause, base_out_dir, holds=predict_jobs)
                            deaths_jobs.append(jid)
                        if 'age_loc_aggregation' in self.processes:
                            jid = self.launch_location_aggregator(
                                year_id, cause_id, int_cause, base_out_dir, holds=deaths_jobs)
                            age_loc_agg_jobs.append(jid)

                    if 'cause_aggregation' in self.processes:
                        jid = self.launch_cause_aggregator(year_id, int_cause, base_out_dir,
                                                           holds=age_loc_agg_jobs)
                        cause_agg_jobs.append(jid)

            if 'compile_rd_props' in self.processes:
                self.launch_compile_rd_props(int_cause, holds=cause_agg_jobs)


if __name__ == '__main__':
    # for help on useage: python intermediate_cause_launch.py --help
    process_order = BurdenCalculator.process_order
    product_list = BurdenCalculator.product_list
    int_causes = BurdenCalculator.valid_intermediate_causes
    parser = argparse.ArgumentParser(description='Launch intermediate cause analyses')
    parser.add_argument(
        '--description', help='model run; if processes="run_model" then date is appended',
        type=str, required=True, nargs=1
    )
    parser.add_argument('--processes', help='processes to be run, e.g. age_loc_aggregation',
                        type=str, required=True, nargs='+', choices=process_order)
    parser.add_argument('--int_causes', help='intermediate cause(s) of interest',
                        required=True, nargs='+', choices=int_causes, type=str)
    parser.add_argument('--end_product', help='how the end result will be used', required=True,
                        nargs=1, choices=product_list, type=str)
    parser.add_argument('--cause_id', help='causes for which to run post-modeling processes',
                        nargs='*', type=int)
    parser.add_argument('--custom', help='Whether or not to use a custom cause hierarchy',
                        type=bool, default=False)
    parser.add_argument('--year_id', help='years for which to run post-modeling processes',
                        nargs='*', default=[1990, 2000, 2010, 2015])
    args = parser.parse_args()
    launcher = BurdenCalculator(vars(args))
    launcher.launch()
