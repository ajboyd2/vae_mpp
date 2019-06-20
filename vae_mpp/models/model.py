from torch.nn import Module

from vae_mpp.utils import Registrable

class Model(Module, Registrable):

    def __init__(self ):
        super().__init__()