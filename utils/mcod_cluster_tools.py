from __future__ import print_function
from builtins import str
import os
import getpass
from cod_prep.claude.configurator import Configurator

CONF = Configurator()


def submit_mcod(
    jobname, language, worker, cores, memory, params=[], holds=[], verbose=False, runtime=None,
    logging=False, log_base_dir=None, new_nodes=False, shell_script=None, jdrive=False,
    queue='all.q', validate_shell=True
):
    """Wrapper for launching a script.

    Constructs a qsub command, appending all *params to the end,
    and calls it using python's subprocess module

    default output:
        /share/temp/sgeoutput/[USER]/output/[JOBNAME]
    default errors:
        /share/temp/sgeoutput/[USER]/errors/[JOBNAME]

    Cluster project defined in cod_prep/claude/claude_config.yaml

    Required Arguments:
        jobname: the name of the job
        language: "python" or "stata"; determines shell_script
        worker: the path to the script
        cores: number of cores
        memory: maximum memory usage, can specify M, G, or T
                (aim to request ~15% more than required)
    Optional Arguments:
        params: dictionary or list of paramaters
        shell_script: optionally pass in a different shell script
        new_nodes: submit jobs to a group of machines that has a new linux kernel
        jdrive: submit jobs to nodes with the J drive mounted
        runtime: maximum run-time (should be ~25% more than what you need)
                even though this is optional, you should use it for the good of the cluster
        holds: list of job ids to wait for before starting
        verbose: whether to print job submission info
        logging: whether to log errors/output

    Returns:
        subprocess return code from calling the command, which returns the parsed job_id
    """
    assert os.path.exists(worker), 'No such file: {}'.format(worker)
    user = getpass.getuser()
    if logging and log_base_dir is None:
        if 'claude' not in jobname:
            base_dir = "/share/temp/sgeoutput"
            log_dir = {
                'output': "{}/{}/output/{}".format(base_dir, user, jobname),
                'errors': "{}/{}/errors/{}".format(base_dir, user, jobname)
            }
        else:
            base_dir = "/ihme/cod/prep/"
            log_dir = {
                'output': "{}/{}/output/{}".format(base_dir, user, jobname),
                'errors': "{}/{}/errors/{}".format(base_dir, user, jobname)
            }
    elif logging and log_base_dir is not None:
        log_dir = {
            'output': "{}/output/{}".format(log_base_dir, jobname),
            'errors': "{}/errors/{}".format(log_base_dir, jobname)
        }
    else:
        log_dir = {
            'output': '/dev/null',
            'errors': '/dev/null'
        }
    if shell_script is None:
        if language == 'python':
            shell_script = "/homes/{u}/cod-data/mcod_prep/" \
                "utils/shell{lng}.sh".format(u=user, lng=language)
        elif language == 'r':
            # https://hub.ihme.washington.edu/display/DataScience/Shell+Scripts
            shell_script = '/ihme/singularity-images/rstudio/shells/execRscript.sh'

    if validate_shell:
        assert os.path.exists(shell_script), 'No such file: {}'.format(shell_script)
    if isinstance(memory, (int, float)):
        # assume that the user inteded gigabytes
        memory = str(memory) + 'G'
    else:
        assert memory[-1] in ['M', 'G', 'T'], \
            'please specify memory as M (megabytes) G (gigabytes) or T (terabytes)'

    return qsub(
        worker, shell_script, CONF.get_id('cluster_project'), cores, memory,
        custom_args=params, name=jobname, holds=holds, runtime=runtime, new_nodes=new_nodes,
        log_dir=log_dir, jdrive=jdrive, verbose=verbose, queue=queue
    )


def qsub(f, shell, proj, cores, memory, custom_args={}, name=None, holds=[], runtime=None,
         log_dir=None, time=None, queue='all.q', jdrive=False, new_nodes=False, verbose=False):
    '''qsub a job that runs a given python, stata, or R script.

    -- Required --
    f: (str) filepath of script to run
    shell: (str) filepath of shell script to run
    proj: (str) project to submit jobs under, e.g. proj_codprep
    cores (int)
    memory (str)
    queue

    -- Optional --
    custom_args : dict or list
        If dictionary, custom arguments for python scripts for use
            with python's argparse library
            Example: {'-ac':'cvd_stroke','-l':102,'-s':1}
        If list, ordered list of arguments
            Example: ['cvd_stroke', '102', '1']
    name : str
        name of job, otherwise uses filename
    holds : list
        list of job id numbers to wait for before running the job
    log_dir : str or dict
        directory in which to store the error and output logs
        Defaults to /dev/null, doing no logging
        Example: log_dir = '/share/temp/sgeoutput/mollieh'
                saves to '/share/temp/sgeoutput/mollieh/errors'
                and '/share/temp/sgeoutput/mollieh/output'

        If log_dir is a dictionary, keys must be either
                ["output","errors"] or ["base_dir","sub_dir"]
        If keys are ['base_dir','sub_dir'], logs will save to
            {base_dir}/{ output or errors }/{sub_dir}
        If keys are ['output','errors'], logs will save to the
            full filepaths specified for each key.
    verbose : boolean, default False
        whether to print job submission info
    time : string in the form MMDDhhmm
        date and time to submit job - For example, Nov 1 at 1am = 11010100
    '''
    # Minimum starting string: job name and project
    if name is None:
        name = f.split('/')[-1]

    qsub_str = 'qsub -N {n} -P {p} -q {q} -l m_mem_free={m} -l fthread={c} '.format(
        p=proj, n=name, q=queue, m=memory, c=cores)

    # Optional qsub arguments
    if log_dir is not None:
        if isinstance(log_dir, str):
            # Remove backslash at end of directory name if it exists
            if log_dir[-1] == '/':
                log_dir = log_dir[0:-1]
            qsub_str += '-o {log_dir}/output -e {log_dir}/errors '.format(
                log_dir=log_dir)
        elif isinstance(log_dir, dict):
            if 'output' in list(log_dir.keys()):
                assert set(log_dir.keys()) == set(['output', 'errors']), \
                    'If log_dir is a dictionary, keys must be either ' \
                    '["output","errors"] or ["base_dir","sub_dir"]'
                qsub_str += '-o {output} -e {errors} '.format(
                    output=log_dir['output'], errors=log_dir['errors'])

            elif 'base_dir' in list(log_dir.keys()):
                assert set(log_dir.keys()) == set(["base_dir", "sub_dir"]), \
                    'If log_dir is a dictionary, keys must be either ' \
                    '["output","errors"] or ["base_dir","sub_dir"]'
                qsub_str += (
                    '-o {base_dir}/output/{sub_dir} -e '
                    '{base_dir}/errors/{sub_dir} '.format(
                        base_dir=log_dir['base_dir'],
                        sub_dir=log_dir['sub_dir']
                    ))

            else:
                raise ValueError(
                    'If log_dir is a dictionary, keys must be either '
                    '["output","errors"] or ["base_dir","sub_dir"]'
                )
        else:
            raise TypeError(
                'log_dir must be either a string or dictionary of strings '
                'with keys ["output","error"] or ["base_dir","sub_dir"]'
            )

    if len(holds) > 0:
        qsub_str += '-hold_jid {holds} '.format(
            holds=','.join([str(h) for h in holds]))
    if new_nodes:
        qsub_str += '-l mlkernel '
    if runtime is not None:
        # !!!TO DO!!!
        # assert that runtime argument is like this: "1:30:00", e.g.
        # or convert it to whatever datetime that is supposed to be
        # can be specified as HH:MM:SS or MM:SS
        qsub_str += '-l h_rt={} '.format(runtime)
    if jdrive:
        qsub_str += '-l archive=TRUE '
    if time is not None:
        qsub_str += '-a {} '.format(time)

    # Always add the shell script and filepath at the end
    if 'R' in shell:
        qsub_str += f'{shell} -s {f}'
    else:
        qsub_str += f'{shell} {f}'

    # Optionally add custom arguments after the script you're running
    if custom_args != {}:
        # For use with python's argparse library
        if isinstance(custom_args, dict):
            for arg in custom_args:
                qsub_str += ' {k} {v}'.format(k=arg, v=custom_args[arg])
        # For ordered list of arguments after script
        elif isinstance(custom_args, list):
            for arg in custom_args:
                qsub_str += ' {}'.format(arg)

    job = os.popen(qsub_str).read()
    if verbose:
        print(job)
    j_id = job.rstrip().split()[2]
    return j_id
