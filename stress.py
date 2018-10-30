from numpy import multiply, divide

class EngineeringStress:
    """Engineering stress is the ratio of force
    and original cross section area."""
    name = 'Engineering stress'

    def from_true_stress(true_stress, stretch):
        """Converts the given true stress to engineering stress.
        ----------
        Keyword arguments:
        true_stress -- True stress to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted engineering stress.
        """
        return divide(true_stress, stretch)

    def to_true_stress(engineering_stress, stretch):
        """Converts the given engineering stress to true stress.
        ----------
        Keyword arguments:
        engineering_stress -- Engineering stress to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted true stress.
        """
        return multiply(stretch, engineering_stress)

class TrueStress:
    """True stress is the ratio of force
    and actual cross section area."""
    name = 'True stress'

    def from_true_stress(true_stress, stretch):
        """Converts the given true stress to true stress.
        ----------
        Keyword arguments:
        true_stress -- True stress to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted true stress.
        """
        return true_stress

    def to_true_stress(true_stress, stretch):
        """Converts the given true stress to true stress.
        ----------
        Keyword arguments:
        true_stress -- True stress to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted true stress.
        """
        return true_stress
