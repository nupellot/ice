from .base import ApproximatorBase
from .temporal import TemporalInterpolator
from .spatial import KrigingInterpolator, DelaunayInterpolator
from .combined import WeightedCombinedInterpolator

# Список всех базовых (не комбинированных) методов
AVAILABLE_APPROXIMATORS = {
    'temporal': TemporalInterpolator,
    'kriging': KrigingInterpolator,
    'delaunay': DelaunayInterpolator,
}

# Добавим комбинированный метод как отдельную опцию
COMBINED_NAME = 'combined'

def create_approximator(name: str, **kwargs) -> ApproximatorBase:
    """
    Создает экземпляр аппроксиматора по имени.
    Если name == 'combined', ожидается аргумент interpolators_dict.

    :param name: имя метода
    :param kwargs: параметры конструктора
    :return: экземпляр аппроксиматора
    """
    if name == COMBINED_NAME:
        return WeightedCombinedInterpolator(**kwargs)
    elif name in AVAILABLE_APPROXIMATORS:
        cls = AVAILABLE_APPROXIMATORS[name]
        return cls(**kwargs)
    else:
        raise ValueError(f"Unknown approximator: '{name}'")

def list_methods(kind: str = None) -> list[str]:
    """
    Возвращает список доступных методов, фильтруя по классу ('temporal', 'spatial').

    :param kind: если задан, фильтрует методы по типу (например 'spatial')
    """
    result = []
    for name, cls in AVAILABLE_APPROXIMATORS.items():
        if kind is None or cls(None, set()).kind == kind:
            result.append(name)
    return result
