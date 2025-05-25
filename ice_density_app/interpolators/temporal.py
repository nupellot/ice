from .base import InterpolatorBase

class TemporalInterpolator(InterpolatorBase):
    """
    Временной интерполятор. Для каждой точки находит ближайшие по времени измерения
    и линейно интерполирует значение, либо возвращает ближайшее доступное, если только оно есть.
    Метрика качества: количество дней до ближайшего наблюдения (чем меньше, тем лучше).
    """

    kind = 'temporal'

    def __init__(self, df, unique_coords):
        """
        df: pandas.DataFrame с колонками [date_only, longitude, latitude, density]
        unique_coords: set с кортежами (lon, lat) всех возможных точек
        """
        super().__init__()
        self.df = df
        self.unique_coords = unique_coords
        # Сгруппируем данные для быстрого доступа
        self.grouped = df.sort_values('date_only').groupby(['longitude', 'latitude'])

    def interpolate_point(self, lon, lat, target_date):
        """
        Интерполирует значение плотности в заданной точке и дате.
        Использует кэш.
        """
        cached = self.get_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        try:
            group = self.grouped.get_group((lon, lat))
        except KeyError:
            return None

        before = group[group['date_only'] < target_date]
        after = group[group['date_only'] > target_date]
        prev_row = before.iloc[-1] if not before.empty else None
        next_row = after.iloc[0] if not after.empty else None

        if prev_row is not None and next_row is not None:
            t0 = prev_row['date_only']
            t1 = next_row['date_only']
            dens0 = prev_row['density']
            dens1 = next_row['density']
            total_days = (t1 - t0).days
            frac = (target_date - t0).days / total_days if total_days > 0 else 0
            density = dens0 + (dens1 - dens0) * frac
        elif prev_row is not None:
            density = prev_row['density']
        elif next_row is not None:
            density = next_row['density']
        else:
            return None

        self.set_to_cache(lon, lat, target_date, float(density))
        return float(density)

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        """
        Рассчитывает значения для всех coords_to_interpolate за указанный target_date.
        Возвращает словарь: {(lon, lat): density, ...}
        """
        interp_results = {}
        known_coords = {(p['longitude'], p['latitude']) for p in known_points}
        for coord in coords_to_interpolate:
            if coord in known_coords:
                continue
            dens = self.interpolate_point(coord[0], coord[1], target_date)
            if dens is not None:
                interp_results[coord] = dens
        return interp_results

    def quality_metric(self, target_date, coord, known_points=None):
        """
        Метрика: минимальное количество дней до ближайшего наблюдения для точки (lon, lat).
        Если данных нет — возвращает None.
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

        # Вычисляем расстояние в днях до ближайшего измерения
        min_days = abs((group['date_only'] - target_date).dt.days).min()
        self.set_quality_to_cache(lon, lat, target_date, min_days)
        return min_days
