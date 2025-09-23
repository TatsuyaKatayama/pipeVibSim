from setuptools import setup, find_packages

setup(
    name='pipeVibSim',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sdynpy @ git+https://github.com/TatsuyaKatayama/sdynpy.git@develop',
        'matplotlib',
    ],
)
