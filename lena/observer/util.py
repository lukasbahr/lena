from lena.observer.lueneberger import LuenebergerObserver
import torch

def get_default_observer(system, params):

    if params['experiment'] == 'autonomous':
        observer = LuenebergerObserver(system.dim_x, system.dim_y)
    else:
        observer = LuenebergerObserver(system.dim_x, system.dim_y, 1)

    observer.set_functions(system)
    observer.set_D(1)
    observer.F = torch.Tensor([[1.0], [1.0], [1.0]])

    return observer