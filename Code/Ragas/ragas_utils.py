from ragas.metrics.result import MetricResult

class DetailedMetricResult(MetricResult):
    """
    A custom MetricResult class that includes traces and a 
    formatted string representation for easier debugging.
    """
    def __repr__(self):
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, traces={self.traces})"

    def __str__(self):
        return (f"Value: {self.value}\n"
                f"Reason: {self.reason}\n"
                f"Traces: {self.traces}")