from llm_benchmark.clients.base import BenchmarkClientBase

class GenAIPerfClient(BenchmarkClientBase):
    """GenAI-Perf benchmark client."""

    def run_single_benchmark(self, test_args, **kwargs):
        raise NotImplementedError("GenAI-Perf client is not yet implemented")
