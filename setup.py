from setuptools import setup, find_packages

setup(
    name='dynamic_utils',
    author='chrisdkolloff',
    author_email='chrisdkolloff@gmail.com',
    description='Utils for dealing with dynamic data (MSM, NMR, etc.)',
    url='https://github.com/chrisdkolloff/dynamic_utils.git',
    version='0.0',
    packages=['dynamic_utils'],
    install_requires=[
        'numpy', 'torch'
    ],
    py_modules=['dynamic_utils'],
)