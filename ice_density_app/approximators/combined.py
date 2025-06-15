from .base import ApproximatorBase

class WeightedCombinedInterpolator(ApproximatorBase):
    """
    Комбинированный аппроксиматор. Объединяет несколько аппроксиматоров с весами.
    Поддерживает авто-взвешивание на основе метрики разброса (стандартного отклонения).
    """

    kind = 'combined'

    def __init__(self, interpolators_dict, weights=None, auto_weights=False):
        super().__init__()
        self.interpolators_dict = interpolators_dict
        self.weights = weights or {name: 1.0 for name in interpolators_dict}
        self.auto_weights = auto_weights

    def approximate(self, target_date, coords_to_interpolate, known_points):
        results_by_method = {
            name: interp.approximate(target_date, coords_to_interpolate, known_points)
            for name, interp in self.interpolators_dict.items()
        }

        final_results = {}
        for coord in coords_to_interpolate:
            numer = 0.0
            denom = 0.0
            dyn_weights = {}

            for name, interp in self.interpolators_dict.items():
                val = results_by_method[name].get(coord)
                if val is None:
                    continue

                if self.auto_weights:
                    std = interp.quality_metric(target_date, coord, known_points)
                    if std is not None and std > 0:
                        dyn_weights[name] = 1.0 / (std + 1e-6)
                    else:
                        dyn_weights[name] = 0.0
                else:
                    dyn_weights[name] = self.weights.get(name, 1.0)

            sum_w = sum(dyn_weights.values())
            if sum_w > 0:
                for name, w in dyn_weights.items():
                    val = results_by_method[name].get(coord)
                    if val is not None:
                        numer += val * w
                        denom += w
                final_value = numer / denom if denom > 0 else None
                if final_value is not None:
                    self.set_to_cache(coord[0], coord[1], target_date, final_value)
                    final_results[coord] = final_value

        return final_results

    def quality_metric(self, target_date, coord, known_points=None):
        stds = []
        ws = []
        for name, interp in self.interpolators_dict.items():
            std = interp.quality_metric(target_date, coord, known_points)
            w = self.weights.get(name, 1.0)
            if std is not None and w > 0:
                stds.append(std)
                ws.append(w)
        if stds and ws:
            return sum(s * w for s, w in zip(stds, ws)) / sum(ws)
        return None
