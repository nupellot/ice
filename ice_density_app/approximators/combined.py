from .base import ApproximatorBase

class WeightedCombinedInterpolator(ApproximatorBase):
    """
    Комбинированный аппроксиматор. Объединяет результаты нескольких аппроксиматоров
    с учетом заданных или автоматически вычисленных весов.
    """

    def __init__(self, interpolators_dict, weights=None, auto_weights=False):
        """
        :param interpolators_dict: словарь {имя_метода: экземпляр_аппроксиматора}
        :param weights: словарь {имя_метода: вес} — пользовательские веса
        :param auto_weights: если True, веса вычисляются автоматически на основе качества
        """
        super().__init__()
        self.interpolators_dict = interpolators_dict
        self.weights = weights or {name: 1.0 for name in interpolators_dict}
        self.auto_weights = auto_weights

    @property
    def kind(self) -> str:
        return 'combined'

    def approximate(self, target_date, coords_to_approximate, known_points):
        """
        Выполняет интерполяцию с комбинированием значений всех подаппроксиматоров.
        """
        results_by_method = {
            name: interp.approximate(target_date, coords_to_approximate, known_points)
            for name, interp in self.interpolators_dict.items()
        }

        combined = {}
        for lon, lat in coords_to_approximate:
            coord = (lon, lat)
            values = {}
            weights_used = {}

            for name, interp in self.interpolators_dict.items():
                value = results_by_method[name].get(coord)
                if value is None:
                    continue

                if self.auto_weights:
                    metric = interp.quality_metric(target_date, coord, known_points)
                    if metric is not None and metric > 0:
                        weights_used[name] = 1.0 / (metric + 1e-6)
                    else:
                        weights_used[name] = 0.0
                else:
                    weights_used[name] = self.weights.get(name, 1.0)

                values[name] = value

            total_weight = sum(weights_used.values())
            if total_weight > 0:
                weighted_sum = sum(values[name] * weights_used[name] for name in values)
                combined_value = weighted_sum / total_weight
                self.set_to_cache(lon, lat, target_date, combined_value)
                combined[coord] = combined_value

        return combined

    def quality_metric(self, target_date, coord, known_points=None):
        """
        Вычисляет средневзвешенную метрику качества среди всех подаппроксиматоров.
        """
        metrics = []
        weights = []

        for name, interp in self.interpolators_dict.items():
            metric = interp.quality_metric(target_date, coord, known_points)
            weight = self.weights.get(name, 1.0)
            if metric is not None:
                metrics.append(metric)
                weights.append(weight)

        if metrics and weights:
            total_weight = sum(weights)
            return sum(m * w for m, w in zip(metrics, weights)) / total_weight

        return None
