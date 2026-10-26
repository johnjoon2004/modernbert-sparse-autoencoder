import torch


def welford_z_update(existing_aggregate, new_values):
    (count, mean, M2) = existing_aggregate
    for x in new_values:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2
    return (count, mean, M2)


def welford_z_finish(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return mean, torch.tensor(float("nan"))
    else:
        (mean, variance) = (mean, M2 / count)
        return mean, torch.sqrt(variance)
