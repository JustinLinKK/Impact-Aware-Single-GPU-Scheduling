"""Two bounded preparation consumers with isolated inference and graph capture."""
from threading import Thread
from ..a10_feedback_latency.async_preparation import PreparedPredictor
from ...scheduler.parallel_preparation import ParallelPreparedPredictor, IsolatedPredictionBackend
from ...prediction.fingerprint_preparation import FingerprintPredictor


class GraphParallelPreparedPredictor(ParallelPreparedPredictor):
    def __init__(self, predictor, *, workers=2, **kwargs):
        graph = getattr(predictor, 'predictor', None)
        backend = getattr(graph, 'predictor', None)
        if (not isinstance(graph, FingerprintPredictor) or not isinstance(backend, IsolatedPredictionBackend)
                or type(workers) is not int or not 1 <= workers <= 2):
            raise ValueError('isolated inference and serialized isolated graph capture required')
        if kwargs.pop('asynchronous', True) is not True or kwargs.pop('prepare_on_read', False):
            raise ValueError('graph preparation must stay asynchronous')
        self._workers, self._backend = [], backend
        PreparedPredictor.__init__(self, predictor, asynchronous=False, prepare_on_read=False, **kwargs)
        self.asynchronous = True
        for i in range(workers):
            thread = Thread(target=self._run, name=f'v511-preparation-{i}', daemon=True)
            self._workers.append(thread); thread.start()
