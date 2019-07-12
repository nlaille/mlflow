import sys
import os
import subprocess
import signal
import logging

from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import get_run_env_vars
from mlflow.entities import RunStatus

_logger = logging.getLogger(__name__)


def _build_mlflow_run_cmd(
        uri, entry_point, storage_dir, use_conda, run_id, parameters):
    """
    Build and return an array containing an ``mlflow run`` command that can be invoked to locally
    run the project at the specified URI.
    """
    mlflow_run_arr = ["mlflow", "run", uri, "-e", entry_point, "--run-id", run_id]
    if storage_dir is not None:
        mlflow_run_arr.extend(["--storage-dir", storage_dir])
    if not use_conda:
        mlflow_run_arr.append("--no-conda")
    for key, value in parameters.items():
        mlflow_run_arr.extend(["-P", "%s=%s" % (key, value)])
    return mlflow_run_arr


def _run_mlflow_run_cmd(mlflow_run_arr, env_map):
    """
    Invoke ``mlflow run`` in a subprocess, which in turn runs the entry point in a child process.
    Returns a handle to the subprocess. Popen launched to invoke ``mlflow run``.
    """
    final_env = os.environ.copy()
    final_env.update(env_map)
    # Launch `mlflow run` command as the leader of its own process group so that we can do a
    # best-effort cleanup of all its descendant processes if needed
    if sys.platform == "win32":
        return subprocess.Popen(
            mlflow_run_arr, env=final_env, universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        return subprocess.Popen(
            mlflow_run_arr, env=final_env, universal_newlines=True, preexec_fn=os.setsid)


def run_mlflow_subprocess(work_dir, entry_point, parameters, active_run,
                          use_conda, storage_dir):
    """
    Run an MLflow project asynchronously by invoking ``mlflow run`` in a subprocess, returning
    a SubmittedRun that can be used to query run status.
    """
    run_id = active_run.info.run_uuid
    _logger.info("=== Asynchronously launching MLflow run with ID %s ===", run_id)
    mlflow_run_arr = _build_mlflow_run_cmd(
        uri=work_dir, entry_point=entry_point, storage_dir=storage_dir, use_conda=use_conda,
        run_id=run_id, parameters=parameters)
    mlflow_run_subprocess = _run_mlflow_run_cmd(
        mlflow_run_arr, get_run_env_vars(active_run))
    return LocalSubmittedRun(run_id, mlflow_run_subprocess)


def run_entry_point(command, active_run, work_dir):
    """
    Run an entry point command in a subprocess, returning a SubmittedRun that can be used to
    query the run's status.
    :param command: Entry point command to run
    :param active_run: Active run to execute
    :param work_dir: Working directory in which to run the command
    """
    run_id = active_run.info.run_uuid
    env = os.environ.copy()
    env.update(get_run_env_vars(active_run))
    _logger.info("=== Running command '%s' in run with ID '%s' === ", command, run_id)
    # in case os name is not 'nt', we are not running on windows. It introduces
    # bash command otherwise.
    if os.name != "nt":
        process = subprocess.Popen(["bash", "-c", command], close_fds=True, cwd=work_dir, env=env)
    else:
        process = subprocess.Popen(command, close_fds=True, cwd=work_dir, env=env)
    return LocalSubmittedRun(run_id, process)


class LocalSubmittedRun(SubmittedRun):
    """
    Instance of ``SubmittedRun`` corresponding to a subprocess launched to run an entry point
    command locally.
    """
    def __init__(self, run_id, command_proc):
        super(LocalSubmittedRun, self).__init__()
        self._run_id = run_id
        self.command_proc = command_proc

    @property
    def run_id(self):
        return self._run_id

    def wait(self):
        return self.command_proc.wait() == 0

    def cancel(self):
        # Interrupt child process if it hasn't already exited
        if self.command_proc.poll() is None:
            # Kill the the process tree rooted at the child if it's the leader of its own process
            # group, otherwise just kill the child
            try:
                if self.command_proc.pid == os.getpgid(self.command_proc.pid):
                    os.killpg(self.command_proc.pid, signal.SIGTERM)
                else:
                    self.command_proc.terminate()
            except OSError:
                # The child process may have exited before we attempted to terminate it, so we
                # ignore OSErrors raised during child process termination
                _logger.info(
                    "Failed to terminate child process (PID %s) corresponding to MLflow "
                    "run with ID %s. The process may have already exited.",
                    self.command_proc.pid, self._run_id)
            self.command_proc.wait()

    def _get_status(self):
        exit_code = self.command_proc.poll()
        if exit_code is None:
            return RunStatus.RUNNING
        if exit_code == 0:
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self):
        return RunStatus.to_string(self._get_status())
