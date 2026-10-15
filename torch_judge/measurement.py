"""Small opt-in timing experiment; correctness belongs to exercise verifiers."""
import statistics
import time
import torch


def measure(operation, *, device='cpu', workload=None, warmup=3, repeats=10):
    """Synchronized wall-clock samples; reports variability and conditions."""
    if warmup < 0 or repeats < 1:
        raise ValueError('warmup must be nonnegative and repeats positive')
    device = torch.device(device)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        operation()
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        operation()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(workload=workload, device=str(device), warmup=warmup,
                repeats=repeats, method='synchronized wall-clock milliseconds',
                samples_ms=samples, median_ms=statistics.median(samples),
                min_ms=min(samples), max_ms=max(samples))
