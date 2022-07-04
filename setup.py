from setuptools import find_packages, setup

setup(
    name="ivf_adc",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)
