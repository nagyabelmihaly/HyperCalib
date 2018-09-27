from math import exp

class EngineeringStrain:
    """Engineering strain is the ratio of length change
    and original length."""
    def __init__(self):
        self.name = 'Engineering strain'

    def get_engineering_strain(self, engineering_strain):
        """Converts the given value to engineering strain."""
        return engineering_strain

class Stretch:
    """Stretch is the ratio of deformed length
    and original length."""
    def __init__(self):
        self.name = 'Stretch'

    def get_engineering_strain(self, stretch):
        """Converts the given value to engineering strain."""
        return stretch - 1

class TrueStrain:
    """True strain is the natural logarithm of the ratio
    of deformed length and original length."""
    def __init__(self):
        self.name = 'True strain'

    def get_engineering_strain(self, true_strain):
        """Converts the given value to engineering strain."""
        return exp(true_strain) - 1