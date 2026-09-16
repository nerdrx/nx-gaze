"""Display geometry and calibration data, in Qt global logical pixels."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Display:
    name: str
    x: int
    y: int
    width: int
    height: int

    def contains(self, x, y):
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def target(self, u, v):
        return self.x + u * (self.width - 1), self.y + v * (self.height - 1)


def signature(displays):
    return [asdict(d) for d in displays]


def targets(displays):
    return [(index, *display.target(u, v)) for index, display in enumerate(displays)
            for u, v in [(0.5, 0.5), (.12, .12), (.88, .12), (.88, .88), (.12, .88)]]


def visible_display(displays, point):
    """Never snap predictions in desktop gaps onto a different monitor."""
    if point is None or not all(math.isfinite(n) for n in point):
        return None
    return next((i for i, d in enumerate(displays) if d.contains(*point)), None)


class Smoother:
    def __init__(self):
        self.point = None
        self.screen = None
        self.time = None

    def reset(self):
        self.point = self.screen = self.time = None

    def update(self, point, screen, now, amount):
        # Frame-rate-independent smoothing; never drag a dot through desktop gaps.
        if self.point is None or screen != self.screen or now - self.time > .35:
            self.point = tuple(point)
        else:
            tau = 2 * max(0, min(1, amount)) ** 2
            alpha = 1 if tau == 0 else 1 - math.exp(-max(0, now - self.time) / tau)
            self.point = tuple(a + alpha * (b - a) for a, b in zip(self.point, point))
        self.time, self.screen = now, screen
        return self.point


class BlinkFilter:
    """Suppress blink frames and a short, stable-open recovery interval."""
    def __init__(self):
        self.recovering = False
        self.open_since = None
        self.open_frames = 0

    def update(self, has_face, blink, now):
        if not has_face:
            self.recovering = False
            self.open_since = None
            self.open_frames = 0
            return False
        if blink:
            self.recovering = True
            self.open_since = None
            self.open_frames = 0
            return True
        if not self.recovering:
            return False
        if self.open_since is None:
            self.open_since = now
        self.open_frames += 1
        if now - self.open_since >= .10 and self.open_frames >= 3:
            self.recovering = False
            return False
        return True


def assess_motion(rows, directions=4):
    """Compare paired unsmoothed errors with equal weight for every direction."""
    import statistics
    groups = [[r for r in rows if r[0] == i] for i in range(directions)]
    if any(len(group) < 6 for group in groups):
        return False, 'Not enough clean samples in every direction.'
    if any(sum(bool(r[5]) for r in group) / len(group) < .8 for group in groups):
        return False, 'Some movements exceeded the learned head range.'
    before = statistics.mean(statistics.median(math.hypot(r[1], r[2]) for r in group) for group in groups)
    after = statistics.mean(statistics.median(math.hypot(r[3], r[4]) for r in group) for group in groups)
    axis_ok = all(statistics.median(abs(r[new]) for r in group) <=
                  max(statistics.median(abs(r[old]) for r in group) * 1.05,
                      statistics.median(abs(r[old]) for r in group) + 5)
                  for group in groups for old, new in [(1, 3), (2, 4)])
    # Improvement must exceed both a relative threshold and tiny pixel noise.
    accepted = after <= before * .9 and before - after >= 3 and axis_ok
    result = f'Moving-head check: {before:.0f} → {after:.0f} px.'
    return accepted, result
