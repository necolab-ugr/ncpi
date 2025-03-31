from setuptools import setup, find_packages

setup(
    name="ncpi",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib"],
    extras_require={
        "rpy2": ["rpy2"]
    }
)