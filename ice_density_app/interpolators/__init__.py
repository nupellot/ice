from .base import InterpolatorBase
from .temporal import TemporalInterpolator
from .spatial import KrigingInterpolator, DelaunayInterpolator
from .combined import WeightedCombinedInterpolator

# Можно также экспортировать список всех доступных типов для автогенерации UI и т.д.
INTERPOLATOR_CLASSES = {
    'temporal': TemporalInterpolator,
    'kriging': KrigingInterpolator,
    'delaunay': DelaunayInterpolator,
    'combined': WeightedCombinedInterpolator
}
