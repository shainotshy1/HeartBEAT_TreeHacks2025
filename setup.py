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
        'pygame>=2.6.1',
        'openai>=1.12.0',
        'scipy>=1.10.0',
        'python-dotenv>=1.0.1',
        'heartpy>=1.2.7',
        'tqdm>=4.67.1',
        'requests>=2.32.3',
    ],
)
