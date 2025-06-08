from collections import defaultdict
from abc import ABC, abstractmethod

class ApproximatorBase(ABC):
    """
    Абстрактный базовый класс для всех аппроксиматоров (в т.ч. интерполяторов).
    Поддерживает кэширование рассчитанных значений и метрик качества.
    """
    kind: str = None
    
    def __init__(self):
        # Кэш аппроксимированных значений: {(lon, lat): {target_date: value, ...}, ...}
        self.value_cache = defaultdict(dict)

        # Кэш метрик качества: {(lon, lat): {target_date: metric, ...}, ...}
        self.quality_cache = defaultdict(dict)

    def get_from_cache(self, lon: float, lat: float, target_date) -> float | None:
        """
        Получить значение из кэша, если оно там есть.
        """
        return self.value_cache.get((lon, lat), {}).get(target_date, None)

    def set_to_cache(self, lon: float, lat: float, target_date, value: float) -> None:
        """
        Сохранить значение в кэш.
        """
        self.value_cache[(lon, lat)][target_date] = value

    def get_quality_from_cache(self, lon: float, lat: float, target_date) -> float | None:
        """
        Получить сохранённую метрику качества из кэша.
        """
        return self.quality_cache.get((lon, lat), {}).get(target_date, None)

    def set_quality_to_cache(self, lon: float, lat: float, target_date, metric: float) -> None:
        """
        Сохранить метрику качества в кэш.
        """
        self.quality_cache[(lon, lat)][target_date] = metric

    @abstractmethod
    def approximate(self, target_date, coords_to_approximate, known_points):
        """
        Основной метод аппроксимации. Возвращает значения для заданных координат и даты.

        :param target_date: объект даты (тип зависит от реализации)
        :param coords_to_approximate: список координат (lon, lat)
        :param known_points: формат зависит от типа аппроксиматора
        :return: dict {(lon, lat): значение}
        """
        pass

    @abstractmethod
    def quality_metric(self, target_date, coord, known_points=None) -> float:
        """
        Оценка качества аппроксимации в заданной точке.

        :param target_date: дата
        :param coord: координаты (lon, lat)
        :param known_points: (опционально) набор известных точек
        :return: числовая метрика (меньше — лучше)
        """
        pass

    @property
    @abstractmethod
    def kind(self) -> str:
        """
        Класс метода: 'temporal' или 'spatial'.
        """
        pass

    @property
    def name(self) -> str:
        """
        Название метода (может переопределяться для фронтенда).
        """
        return self.__class__.__name__
