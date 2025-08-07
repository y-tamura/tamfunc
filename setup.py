from setuptools import setup, find_packages

setup(
    name='tamfunc',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'xarray',
        'numpy',
        'dask',
        'cftime',
        'scipy'
    ],
    author='y-tamura',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/y-tamura/tamfunc',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
