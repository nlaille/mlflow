from mlflow.store.file_store import FileStore


class DockerFileStore(FileStore):
    def _get_run_info(self, run_uuid):
        run_info = super(DockerFileStore, self)._get_run_info(run_uuid)
        return run_info