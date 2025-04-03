from setuptools import setup, find_packages

setup(
    name="ncpi",
    version="0.1",
    author="necolab",
    email="pablomc@ugr.es",
    description="Neural circuit parameter inference using electrophysiological data",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pycatch22",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib"],
    extras_require={
        "torch": ["torch"],
        "LFPy": ["LFPy"],
        "lfpykernels": ["lfpykernels"],
        "lfpykit": ["lfpykit"],
        "neuron": ["neuron"],
        "nest": ["nest"],
        "fooof": ["fooof"],
        "h5py": ["h5py"],
        "pathos": ["pathos"],
        "tqdm": ["tqdm"],
        "matlab": ["matlab"],
        "matlabengine": ["matlabengine"],
        "rpy2": ["rpy2"],
        "mpl_toolkits": ["mpl_toolkits"],
        "PyAstronomy": ["PyAstronomy"],
    }
)