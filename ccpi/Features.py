import importlib
import subprocess

class Features:
    """
    Class for computing features from electrophysiological data recordings.
    """
    def __init__(self, method = 'catch22'):
        """
        Constructor method.

        Parameters
        ----------
        method: str
            Method to compute features. Default is 'catch22'.
        """
        self.method = method

        # Check if the method is valid
        if self.method not in ['catch22']:
            raise ValueError("Invalid method. Please use 'catch22'.")


    def install(self,module):
        """
        Function to install a Python module.

        Parameters
        ----------
        module: str
            Module name.
        """
        subprocess.check_call(['pip', 'install', module])
        print(f"The module {module} was installed!")


    def catch22(self,sample):
        """
        Function to compute the catch22 features.

        Parameters
        ----------
        sample: np.array
            Sample data.

        Returns
        -------
        features: np.array
            Array with the catch22 features.
        """

        # Dynamically import the pycatch22 module
        try:
            pycatch22 = importlib.import_module('pycatch22')
        except ImportError:
            print("pycatch22 is not installed!")
            self.install('pycatch22')

        features = pycatch22.catch22_all(sample)
        return features['values']