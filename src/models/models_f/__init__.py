import os
import pkgutil
import importlib

__all__ = []

# Перебираем все модули в текущем пакете
for module_info in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_info.name}", package=__name__)
    # Перебираем атрибуты импортированного модуля
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        # Если атрибут является классом и определён в модуле
        if isinstance(attr, type) and attr.__module__ == module.__name__:
            globals()[attr_name] = attr
            __all__.append(attr_name)
