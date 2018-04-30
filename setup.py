from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow_gpu==1.7', 'numpy==1.14.1', 'Pillow==5.1.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True)
