from abc import ABC, abstractmethod
from collections import defaultdict

class ApproximatorBase(ABC):
    """
    Базовый класс для всех аппроксиматоров (временных, пространственных и комбинированных).
    Поддерживает кэш значений и метрик качества, где метрика интерпретируется как ожидаемое стандартное отклонение (разброс).
    """

    def __init__(self):
        # Кэш аппроксимированных значений: {(lon, lat): {target_date: value}}
        self.approx_cache = defaultdict(dict)

        # Кэш оценок качества (разброса): {(lon, lat): {target_date: stddev}}
        self.quality_cache = defaultdict(dict)

    def get_from_cache(self, lon, lat, target_date):
        return self.approx_cache.get((lon, lat), {}).get(target_date, None)

    def set_to_cache(self, lon, lat, target_date, value):
        self.approx_cache[(lon, lat)][target_date] = value

    def get_quality_from_cache(self, lon, lat, target_date):
        return self.quality_cache.get((lon, lat), {}).get(target_date, None)

    def set_quality_to_cache(self, lon, lat, target_date, value):
        self.quality_cache[(lon, lat)][target_date] = value

    @abstractmethod
    def approximate(self, target_date, coords_to_interpolate, known_points):
        """
        Вычисляет значения аппроксимации для заданных координат и даты.
        :param target_date: дата аппроксимации
        :param coords_to_interpolate: список координат (lon, lat)
        :param known_points: известные данные (обычно список dict с ключами: longitude, latitude, density)
        :return: словарь {(lon, lat): значение}
        """
        pass

    @abstractmethod
    def quality_metric(self, target_date, coord, known_points=None):
        """
        Возвращает оценку разброса аппроксимации (ожидаемое стандартное отклонение).
        Чем меньше — тем лучше.

        :param target_date: дата аппроксимации
        :param coord: координаты (lon, lat)
        :param known_points: список известных точек
        :return: разброс (stddev), float
        """
        pass

    @property
    @abstractmethod
    def kind(self):
        """
        Тип метода: 'temporal', 'spatial' или 'combined'
        """
        pass
