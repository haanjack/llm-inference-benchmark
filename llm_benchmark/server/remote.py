import logging

from llm_benchmark.server.base import BenchmarkBase

logger = logging.getLogger(__name__)


class RemoteServer(BenchmarkBase):
    """A 'dummy' server class for benchmarking a remote endpoint."""

    def __init__(self, endpoint: str, **kwargs):
        # We don't need a test plan to start the server, but the base class might use it.
        # Let's remove it from kwargs if it exists to avoid errors.
        self._endpoint = endpoint
        kwargs.pop("test_plan", None)
        super().__init__(**kwargs)

    def start(self):
        """Does nothing, as the server is already running remotely."""
        logger.info("Connecting to remote server at %s", self._endpoint)

    def stop(self):
        """Does nothing, as we don't manage the remote server's lifecycle."""
        pass

    def cleanup(self):
        """Does nothing, as there are no local resources to clean up."""
        pass

    @property
    def endpoint(self) -> str:
        """Returns the endpoint."""
        return self._endpoint