from numpy import exp, log

class EngineeringStrain:
    """Engineering strain is the ratio of length change
    and original length."""
    name = 'Engineering strain'

    def from_stretch(stretch):
        """Converts the given stretch to engineering strain."""
        return stretch - 1

    def to_stretch(engineering_strain):
        """Converts the given engineering strain to stretch."""
        return engineering_strain + 1

class Stretch:
    """Stretch is the ratio of deformed length
    and original length."""
    name = 'Stretch'

    def from_stretch(stretch):
        """Converts the given stretch to stretch."""
        return stretch

    def to_stretch(stretch):
        """Converts the given stretch to stretch."""
        return stretch

class TrueStrain:
    """True strain is the natural logarithm of the ratio
    of deformed length and original length."""
    name = 'True strain'

    def from_stretch(stretch):
        """Converts the given stretch to engineering strain."""
        return log(stretch)

    def to_stretch(true_strain):
        """Converts the given true strain to stretch."""
        return exp(true_strain)