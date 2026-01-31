import numpy as np
class IdentityCal:
    def predict(self, s):
        s=np.asarray(s).reshape(-1)
        return np.clip(s, 1e-6, 1-1e-6)
