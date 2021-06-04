from setuptools import setup

setup(
    name='lena',
    url='https://github.com/lukasbahr/lena',
    author='Lukas Bahr',
    packages=['lnos.net', 'lnos.datasets', 'lnos.observer'],
    install_requires=['numpy', 'torch', 'scipy', 'matplotlib', 'torchdiffeq', 'smt'],
    version='0.01',
    license='MIT',
    description='Implementation of lnos',
)
