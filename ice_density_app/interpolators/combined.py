from .base import InterpolatorBase

class WeightedCombinedInterpolator(InterpolatorBase):
    """
    Класс-комбинатор: объединяет несколько интерполяторов с весами.
    Может принимать веса явно или вычислять их на основе качества каждого метода.
    """
    kind = 'combined'

    def __init__(self, interpolators_dict, weights=None, auto_weights=False):
        """
        interpolators_dict: dict {name: interpolator}
        weights: dict {name: число} — пользовательские веса (по умолчанию одинаковые)
        auto_weights: bool — если True, веса считаются автоматически на основе quality_metric (меньше — лучше)
        """
        super().__init__()
        self.interpolators_dict = interpolators_dict
        self.weights = weights or {name: 1.0 for name in interpolators_dict}
        self.auto_weights = auto_weights

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        """
        Вычисляет значения всех интерполяторов, затем усредняет по весам для каждой точки.
        """
        # Сначала получим результаты каждого метода
        results = {}
        for name, interpolator in self.interpolators_dict.items():
            results[name] = interpolator.interpolate(target_date, coords_to_interpolate, known_points)

        combined = {}
        for coord in coords_to_interpolate:
            numer = 0.0
            denom = 0.0
            dynamic_weights = {}

            # Для каждой точки можно либо использовать фиксированные веса, либо динамические (по метрике)
            for name, interpolator in self.interpolators_dict.items():
                value = results[name].get(coord)
                # Пропускаем если не рассчитано
                if value is None:
                    continue

                if self.auto_weights:
                    # Получаем метрику качества (чем меньше — тем лучше)
                    metric = interpolator.quality_metric(target_date, coord, known_points)
                    # Простая схема: вес = 1 / (metric + eps), если метрика есть
                    if metric is not None and metric > 0:
                        dynamic_weights[name] = 1.0 / (metric + 1e-6)
                    else:
                        dynamic_weights[name] = 0.0
                else:
                    dynamic_weights[name] = self.weights.get(name, 1.0)

            # Суммируем по весам
            sum_weights = sum(dynamic_weights.values())
            if sum_weights > 0:
                for name, interpolator in self.interpolators_dict.items():
                    value = results[name].get(coord)
                    if value is not None:
                        w = dynamic_weights[name]
                        numer += value * w
                        denom += w
                combined[coord] = numer / denom if denom > 0 else None
                self.set_to_cache(coord[0], coord[1], target_date, combined[coord])
            # Если нет ни одного результата — ничего не пишем

        return combined

    def quality_metric(self, target_date, coord, known_points=None):
        """
        Для комбинированного метода возвращаем средневзвешенное качество всех подметодов.
        """
        metrics = []
        weights = []
        for name, interpolator in self.interpolators_dict.items():
            m = interpolator.quality_metric(target_date, coord, known_points)
            w = self.weights.get(name, 1.0)
            if m is not None:
                metrics.append(m)
                weights.append(w)
        if metrics and weights:
            return sum(m * w for m, w in zip(metrics, weights)) / sum(weights)
        return None
