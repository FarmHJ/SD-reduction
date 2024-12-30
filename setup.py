#
# setuptools script
#
from setuptools import setup, find_packages


setup(
    # Module name (lowercase)
    name='modelling',

    # Version
    version='0.0.0',

    description='Code for analysis of the SD drug-binding formulation.',  # noqa

    # long_description=get_readme(),

    license='BSD 3-Clause "New" or "Revised" License',

    # author='',

    # author_email='',

    maintainer='',

    maintainer_email='',

    # url='',

    # Packages to include
    packages=find_packages(include=('modelling', 'modelling.*')),

    # List of dependencies
    install_requires=[
        # Dependencies go here!
        'matplotlib',
        'myokit',
        'numpy>=1.8',
        'pandas',
    ],
    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
            # Nice theme for docs
            'sphinx_rtd_theme',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
        ],
    },
)
