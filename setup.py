# This Python file uses the following encoding: utf-8
from setuptools import setup, find_packages

setup(  
    name='kilograms',
    packages=find_packages(),
    version='0.0.3',
    description='Histograms for large data.',
    long_description='Numba based calculation of 1D and 2D histograms for large data..',
    author='MatteoLacki',
    author_email='matteo.lacki@gmail.com',
    keywords=['Great module', 'Devel Inside'],
    classifiers=['Development Status :: 1 - Planning',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 ],
    install_requires=[
        "numba",
        "numpy",
        "matplotlib",
        "pandas",# can remove it later
    ],
    scripts=[
        #"bin/run4DFF.py",
    ]
)
