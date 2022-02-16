import numpy as np

"""In all functions, we assume that `true_cps` and `reported_cps` are sorted from lowest to highest."""


def percent_changes_detected(true_cps, reported_cps):
    return len(reported_cps) / len(true_cps) * 100


def mean_until_detection(true_cps, reported_cps):
    reported_cps = reported_cps.copy()
    dist = 0
    detected_cps = 0
    for cpi, true_cp in enumerate(true_cps):
        next_cp = true_cps[cpi + 1] if cpi < len(true_cps) - 1 else np.infty
        for reported_cp in reported_cps:
            if reported_cp <= true_cp:
                continue
            if reported_cp >= next_cp:
                continue
            dist += reported_cp - true_cp
            detected_cps += 1
            reported_cps.remove(reported_cp)
            break
    return dist / detected_cps if detected_cps > 0 else np.nan


def true_positives(true_cps, reported_cps, T=10):
    true_cps = true_cps.copy()
    tps = 0
    for reported_cp in reported_cps:
        for true_cp in true_cps:
            if abs(true_cp - reported_cp) < T:
                tps += 1
                true_cps.remove(true_cp)
                break
    return tps


def false_positives(true_cps, reported_cps, T=10):
    tps = true_positives(true_cps, reported_cps, T)
    print(tps)
    return len(reported_cps) - tps


def false_negatives(true_cps, reported_cps, T=10):
    reported_cps = reported_cps.copy()
    fns = len(true_cps)
    for true_cp in true_cps:
        for reported_cp in reported_cps:
            if abs(true_cp - reported_cp) < T:
                fns -= 1
                reported_cps.remove(reported_cp)
                break
    return fns


def precision(tp, fp, fn):
    if tp + fp == 0:
        return np.nan
    return tp / (tp + fp)


def recall(tp, fp, fn):
    if tp + fn == 0:
        return np.nan
    return tp / (tp + fn)


def fb_score(true_cps, reported_cps, T=10, beta=1):
    tps = true_positives(true_cps, reported_cps, T)
    fps = false_positives(true_cps, reported_cps, T)
    fns = false_negatives(true_cps, reported_cps, T)
    prec = precision(tps, fps, fns)
    rec = recall(tps, fps, fns)
    return (1 + beta ** 2) * (prec * rec) / ((beta ** 2 * prec) + rec)


def test_true_positives():
    true_cps = [99, 200, 400]
    reported_cps = [100, 150, 200, 250, 300, 400, 500]
    assert true_positives(true_cps, reported_cps) == 3


def test_true_positives2():
    true_cps = [99, 200, 400]
    reported_cps = [50, 400]
    assert true_positives(true_cps, reported_cps) == 1


def test_true_positives3():
    true_cps = [99, 102]
    reported_cps = [50, 100, 200]
    assert true_positives(true_cps, reported_cps) == 1


def test_true_positives4():
    true_cps = [70, 102]
    reported_cps = [100, 101]
    assert true_positives(true_cps, reported_cps) == 1


def test_false_negatives():
    true_cps = [99, 102]
    reported_cps = [50]
    assert false_negatives(true_cps, reported_cps) == 2


def test_false_negatives2():
    true_cps = [99, 102]
    reported_cps = [50, 100]
    assert false_negatives(true_cps, reported_cps) == 1


def test_false_negatives3():
    true_cps = [102]
    reported_cps = [50, 100, 101]
    assert false_negatives(true_cps, reported_cps) == 0


def test_fb():
    true_cps = [102]
    reported_cps = [50, 100, 101]
    tps = true_positives(true_cps, reported_cps)
    fps = false_positives(true_cps, reported_cps)
    fns = false_negatives(true_cps, reported_cps)
    assert fb_score(true_cps, reported_cps) == tps / (tps + 0.5 * (fps + fns))


def test_mean_until_detection():
    true_cps = [100]
    reported_cps = [101]
    assert mean_until_detection(true_cps, reported_cps) == 1


def test_mean_until_detection2():
    true_cps = [100, 200]
    reported_cps = [101, 204]
    assert mean_until_detection(true_cps, reported_cps) == 5 / 2


def test_mean_until_detection3():
    true_cps = [100, 200]
    reported_cps = [101, 150, 160, 180, 210]
    assert mean_until_detection(true_cps, reported_cps) == 11 / 2
