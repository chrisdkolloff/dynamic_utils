from setuptools import setup, find_packages

setup(
    name='dynamic_utils',
    author='chrisdkolloff',
    author_email='chrisdkolloff@gmail.com',
    description='Utils for dealing with dynamic data (MSM, NMR, etc.)',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 'torch'
    ],
)