from __future__ import absolute_import
import logging
import docker
import time
import os
import yaml
from threading import RLock
import kubernetes
from datetime import datetime

from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus
from mlflow import tracking

_logger = logging.getLogger(__name__)


def before_run_validations(tracking_uri, backend_config):
    """Validations to perform before running a project on Kubernetes."""
    if backend_config is None:
        raise ExecutionException("Backend spec must be provided when launching MLflow project "
                                 "runs on Kubernetes.")
    # TODO: which tracking url are supported
    if not tracking.utils._is_databricks_uri(tracking_uri) and \
            not tracking.utils._is_http_uri(tracking_uri):
        raise ExecutionException(
            "When running on Databricks, the MLflow tracking URI must be of the form "
            "'databricks' or 'databricks://profile', or a remote HTTP URI accessible to both the "
            "current client and code running on Databricks. Got local tracking URI %s. "
            "Please specify a valid tracking URI via mlflow.set_tracking_uri or by setting the "
            "MLFLOW_TRACKING_URI environment variable." % tracking_uri)


def validate_project_execution(project):
    if not project.docker_env:
        raise ExecutionException(
            "Only docker-based projects are supported on Kubernetes.")


def parse_kubernetes_config(backend_config):
    """
    Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
    """
    if not backend_config:
        raise ExecutionException("Backend_config file not found.")
    kube_config = backend_config.copy()
    if 'kube-job-template-path' not in backend_config.keys():
        raise ExecutionException("'kube-job-template-path' attribute must be specified in "
                                 "backend_config.")
    kube_job_template = backend_config['kube-job-template-path']
    if os.path.exists(kube_job_template):
        with open(kube_job_template, 'r') as job_template:
            yaml_obj = yaml.safe_load(job_template.read())
        kube_job_template = yaml_obj
        kube_config['kube-job-template'] = kube_job_template
    else:
        raise ExecutionException("Could not find 'kube-job-template-path': {}".format(
            kube_job_template))
    if 'kube-context' not in backend_config.keys():
        raise ExecutionException("Could not find kube-context in backend_config.")
    if 'image-uri' not in backend_config.keys():
        raise ExecutionException("Could not find 'image-uri' in backend_config.")
    return kube_config


def push_image_to_registry(image_tag):
    client = docker.from_env()
    _logger.info("=== Pushing docker image %s ===", image_tag)
    for line in client.images.push(repository=image_tag, stream=True, decode=True):
        if 'error' in line and line['error']:
            raise ExecutionException("Error while pushing to docker registry: "
                                     "{error}".format(error=line['error']))
    return client.images.get_registry_data(image_tag).id


def _get_kubernetes_job_definition(project_name, image_tag, image_digest,
                                   command, env_vars, job_template):
    container_image = image_tag + '@' + image_digest
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    job_name = "{}-{}".format(project_name, timestamp)
    _logger.info("=== Creating Job %s ===", job_name)
    environment_variables = [{'name': k, 'value': v} for k, v in env_vars.items()]
    job_template['metadata']['name'] = job_name
    job_template['spec']['template']['spec']['containers'][0]['name'] = project_name
    job_template['spec']['template']['spec']['containers'][0]['image'] = container_image
    job_template['spec']['template']['spec']['containers'][0]['command'] = command
    if 'env' not in job_template['spec']['template']['spec']['containers'][0].keys():
        job_template['spec']['template']['spec']['containers'][0]['env'] = []
    job_template['spec']['template']['spec']['containers'][0]['env'] += environment_variables
    return job_template


def _get_run_command(entrypoint_command):
    formatted_command = []
    for cmd in entrypoint_command:
        formatted_command = cmd.split(" ")
    return formatted_command


def run_kubernetes_job(project_name, active_run, image_tag, image_digest, command, env_vars,
                       kube_context, job_template=None):
    job_template = _get_kubernetes_job_definition(project_name,
                                                  image_tag,
                                                  image_digest,
                                                  _get_run_command(command),
                                                  env_vars,
                                                  job_template)
    job_name = job_template['metadata']['name']
    job_namespace = job_template['metadata']['namespace']
    kubernetes.config.load_kube_config(context=kube_context)
    api_instance = kubernetes.client.BatchV1Api()
    api_instance.create_namespaced_job(namespace=job_namespace,
                                       body=job_template, pretty=True)
    return KubernetesSubmittedRun(active_run.info.run_id, job_name, job_namespace)


class KubernetesSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Kubernetes job name.
    :param job_namespace: Kubernetes job namespace.
    """
    # How often to poll run status when waiting on a run
    POLL_STATUS_INTERVAL = 5

    def __init__(self, mlflow_run_id, job_name, job_namespace):
        super(KubernetesSubmittedRun, self).__init__()
        self._mlflow_run_id = mlflow_run_id
        self._job_name = job_name
        self._job_namespace = job_namespace
        self._status = RunStatus.SCHEDULED
        self._status_lock = RLock()

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        kube_api = kubernetes.client.BatchV1Api()
        while not RunStatus.is_terminated(self._update_status(kube_api)):
            time.sleep(self.POLL_STATUS_INTERVAL)

        return self._status == RunStatus.FINISHED

    def _update_status(self, kube_api=kubernetes.client.BatchV1Api()):
        api_response = kube_api.read_namespaced_job_status(name=self._job_name,
                                                           namespace=self._job_namespace,
                                                           pretty=True)
        status = api_response.status
        with self._status_lock:
            if RunStatus.is_terminated(self._status):
                return self._status
            if self._status == RunStatus.SCHEDULED:
                if api_response.status.start_time is None:
                    _logger.info("Waiting for Job to start")
                else:
                    _logger.info("Job started.")
                    self._status = RunStatus.RUNNING
            if status.conditions is not None:
                for condition in status.conditions:
                    if condition.status == "True":
                        _logger.info(condition.message)
                        if condition.type == "Failed":
                            self._status = RunStatus.FAILED
                        elif condition.type == "Complete":
                            self._status = RunStatus.FINISHED
        return self._status

    def get_status(self):
        status = self._status
        return status if RunStatus.is_terminated(status) else self._update_status()

    def cancel(self):
        with self._status_lock:
            if not RunStatus.is_terminated(self._status):
                _logger.info("Cancelling job.")
                kube_api = kubernetes.client.BatchV1Api()
                kube_api.delete_namespaced_job(name=self._job_name,
                                               namespace=self._job_namespace,
                                               body=kubernetes.client.V1DeleteOptions(),
                                               pretty=True)
                self._status = RunStatus.KILLED
                _logger.info("Job cancelled.")
            else:
                _logger.info("Attempting to cancel a job that is already terminated.")
