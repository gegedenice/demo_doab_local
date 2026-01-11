class MockMetric:
    def compute(self, prediction: str, reference: str) -> float:
        if not reference:
            return 0.0
        if prediction == reference:
            return 1.0
        return 0.5
