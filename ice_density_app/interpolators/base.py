from collections import defaultdict
from abc import ABC, abstractmethod

class InterpolatorBase(ABC):
    """
    Абстрактный базовый класс для всех интерполяторов (пространственных, временных, комбинированных).
    Поддерживает кэширование рассчитанных значений для ускорения повторных вычислений.
    """

    def __init__(self):
        # Кэш: {(lon, lat): {date: значение, ...}, ...}
        self.interp_cache = defaultdict(dict)
        # Кэш для метрик качества (по тому же принципу)
        self.quality_cache = defaultdict(dict)

    def get_from_cache(self, lon, lat, target_date):
        """
        Получить интерполированное значение из кэша, если он есть.
        """
        return self.interp_cache.get((lon, lat), {}).get(target_date)

    def set_to_cache(self, lon, lat, target_date, value):
        """
        Сохранить интерполированное значение в кэш.
        """
        self.interp_cache[(lon, lat)][target_date] = value

    def get_quality_from_cache(self, lon, lat, target_date):
        """
        Получить сохранённую метрику качества из кэша.
        """
        return self.quality_cache.get((lon, lat), {}).get(target_date)

    def set_quality_to_cache(self, lon, lat, target_date, metric):
        """
        Сохранить метрику качества в кэш.
        """
        self.quality_cache[(lon, lat)][target_date] = metric

    @abstractmethod
    def interpolate(self, target_date, coords_to_interpolate, known_points):
        """
        Основной метод интерполяции.
        Возвращает словарь: {(lon, lat): значение, ...}
        """
        pass

    @abstractmethod
    def quality_metric(self, target_date, coord, known_points=None):
        """
        Метод оценки качества интерполяции для одной точки.
        Например: кригинговская дисперсия, расстояние до ближайшей точки и т.п.
        Возвращает число (меньше — лучше качество).
        """
        pass

    @property
    @abstractmethod
    def kind(self):
        """
        Тип интерполяции: 'temporal' (временная) или 'spatial' (пространственная).
        """
        pass
