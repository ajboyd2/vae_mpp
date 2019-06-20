from torch.nn import Module

from vae_mpp.utils import Registerable

class Model(Module, Registerable):

    def __init__(self ):
        super().__init__()