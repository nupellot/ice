from .base import ApproximatorBase

class TemporalInterpolator(ApproximatorBase):
    """
    Временной интерполятор. Для каждой точки находит ближайшие по времени измерения
    и линейно интерполирует значение, либо возвращает ближайшее доступное, если только оно есть.

    Метрика качества: количество дней до ближайшего наблюдения (чем меньше, тем лучше).
    """

    def __init__(self, df, unique_coords):
        """
        :param df: pandas.DataFrame с колонками [date_only, longitude, latitude, density]
        :param unique_coords: множество кортежей (lon, lat) — все координаты, которые можно интерполировать
        """
        super().__init__()
        self.df = df
        self.unique_coords = unique_coords

        # Предварительно группируем данные по координатам и сортируем по времени
        self.grouped = df.sort_values('date_only').groupby(['longitude', 'latitude'])

    @property
    def kind(self) -> str:
        return 'temporal'

    def approximate(self, target_date, coords_to_approximate, known_points):
        """
        Выполняет интерполяцию для каждой точки в списке coords_to_approximate.
        Пропускает координаты, уже присутствующие в known_points.

        :param target_date: дата, на которую интерполируем
        :param coords_to_approximate: список координат [(lon, lat), ...]
        :param known_points: список словарей с ключами [longitude, latitude]
        :return: словарь {(lon, lat): плотность}
        """
        interp_results = {}
        known_coords = {(p['longitude'], p['latitude']) for p in known_points}

        for lon, lat in coords_to_approximate:
            if (lon, lat) in known_coords:
                continue

            value = self.get_from_cache(lon, lat, target_date)
            if value is None:
                value = self._interpolate_point(lon, lat, target_date)
                if value is not None:
                    self.set_to_cache(lon, lat, target_date, value)

            if value is not None:
                interp_results[(lon, lat)] = value

        return interp_results

    def _interpolate_point(self, lon, lat, target_date):
        """
        Интерполяция плотности льда для одной точки. Используется внутренне.
        """
        try:
            group = self.grouped.get_group((lon, lat))
        except KeyError:
            return None

        before = group[group['date_only'] < target_date]
        after = group[group['date_only'] > target_date]
        prev_row = before.iloc[-1] if not before.empty else None
        next_row = after.iloc[0] if not after.empty else None

        if prev_row is not None and next_row is not None:
            # Линейная интерполяция между ближайшими срезами
            t0, t1 = prev_row['date_only'], next_row['date_only']
            d0, d1 = prev_row['density'], next_row['density']
            total_days = (t1 - t0).days
            frac = (target_date - t0).days / total_days if total_days > 0 else 0
            return float(d0 + (d1 - d0) * frac)

        elif prev_row is not None:
            return float(prev_row['density'])

        elif next_row is not None:
            return float(next_row['density'])

        return None

    def quality_metric(self, target_date, coord, known_points=None) -> float | None:
        """
        Оценивает качество интерполяции: чем меньше расстояние по времени до ближайшего измерения, тем лучше.

        :param target_date: дата интерполяции
        :param coord: кортеж (lon, lat)
        :return: число дней до ближайшего наблюдения (или None, если данных нет)
        """
        lon, lat = coord

        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        try:
            group = self.grouped.get_group((lon, lat))
        except KeyError:
            return None

        if group.empty:
            return None

        # Рассчитываем минимальное расстояние в днях до ближайшего измерения
        delta_days = abs((group['date_only'] - target_date).dt.days)
        min_days = delta_days.min()

        self.set_quality_to_cache(lon, lat, target_date, min_days)
        return min_days
