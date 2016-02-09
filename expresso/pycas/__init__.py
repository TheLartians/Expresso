
import printer

from .expression import *
from .functions import *
from .compilers import *
from .algorithms import *


def use_global_cache(v):
    from .evaluators import use_global_cache
    use_global_cache(v)

