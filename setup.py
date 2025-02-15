from setuptools import setup, find_packages

setup(
    name="heartbeat",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pyserial>=3.5',
        'numpy>=1.24.0',
        'pandas>=1.5.2',
        'matplotlib>=3.6.2',
        'ipykernel>=6.23.1',
        'scipy>=1.10.0',
    ],
)
