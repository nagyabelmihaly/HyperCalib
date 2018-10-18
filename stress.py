class EngineeringStress:
    """Engineering stress is the ratio of force
    and original cross section area."""
    def __init__(self):
        self.name = 'Engineering stress'

    def get_engineering_stress(self, engineering_stress, stretch):
        """Converts the given stress value to engineering stress.
        ----------
        Keyword arguments:
        engineering_stress -- Stress value to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted engineering stress.
        """
        return engineering_stress

    def get_true_stress(self, engineering_stress, stretch):
        """Converts the given stress value to true stress.
        ----------
        Keyword arguments:
        engineering_stress -- Stress value to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted true stress.
        """
        return stretch * engineering_stress

class TrueStress:
    """True stress is the ratio of force
    and actual cross section area."""
    def __init__(self):
        self.name = 'True stress'

    def get_engineering_stress(self, true_stress, stretch):
        """Converts the given stress value to engineering stress.
        ----------
        Keyword arguments:
        true_stress -- Stress value to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted engineering stress.
        """
        return true_stress / stretch

    def get_true_stress(self, true_stress, stretch):
        """Converts the given stress value to true stress.
        ----------
        Keyword arguments:
        true_stress -- Stress value to convert.
        stretch -- The ratio of deformed length and original length.
        ----------
        Returns:
        The value of converted true stress.
        """
        return true_stress
