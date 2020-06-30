import joblib


class ReplayBufferBase:

    def save(self, fn) -> None:
        joblib.dump(self, fn, compress=3)

    @classmethod
    def load(cls, fn: str) -> "ReplayBufferBase":
        return joblib.load(fn)
