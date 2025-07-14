from setuptools import setup, find_packages

setup(
    name='pipeVibSim',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sdynpy',
        'matplotlib',
    ],
)
