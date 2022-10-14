

class CUSUM():
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.refPoint = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.refPoint += sample / self.M
            return 0
        else:
            s_plus = (sample - self.refPoint) - self.eps
            s_minus = -(sample - self.refPoint) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t = 0
        self.refPoint = 0
        self.g_plus = 0
        self.g_minus = 0
