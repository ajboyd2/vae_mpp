from collections import defaultdict


class Registrable:
    _registry = defaultdict(dict)

    @classmethod
    def register(cls, name):
        registry = Registrable._registry[cls]
        def add_subclass_to_registry(subclass):
            if name in registry:
                raise ValueError('"%s" already registered' % name)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def get(cls, name):
        if name not in Registrable._registry[cls]:
            raise ValueError('"%s" not registered' % name)
        return Registrable._registry[cls].get(name)

    @classmethod
    def from_config(cls, config, **kwargs):
        constructor = cls.get(config.pop('name'))
        return constructor(**config, **kwargs)
