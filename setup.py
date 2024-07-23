# Using setuptools for installation
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # This is the name of the project. 
    # It will determine how users can install this project, e.g.:
    #
    # $ pip install twincausal
    #
    # And where it will live on PyPI: https://pypi.org/project/twincausal/

    name='twincausal',  # Required

    # Versions comply with PEP 440:
    # twincausal version convention, ("major.minor.micro")

    version='0.1.7',
    description='Twin Neural Networks for Uplift',
    license='MIT License',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/belbahrim/twin-causal-net',
    author='Belbahri, M., Gandouet, O., and Sharoff. P',
    author_email=['mouloud.belbahri@gmail.com'],

    # Classifiers help users find the project by categorizing it.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='uplift, deeplearning, machinelearning',
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6',

    # This field lists other packages that this project depends on to run.
    # Any package in the list will be installed by pip when the project is
    # installed, so they must be valid existing projects.
    install_requires=['numpy>=1.19.2',
                      'scikit-learn>=0.23.2',
                      'pandas>=1.1.3',
                      'matplotlib>=3.3.2',
                      'scipy>=1.5',
                      'tensorboard>=2.3.0',
                      'tensorboardX>=1.9',
                      'tensorboard-plugin-wit>=1.7.0',
                      # 'yaml>=0.2.5'
                      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install twincausal[dev]
    #
    # Similar to `install_requires` above, these must be valid existing projects.
    extras_require={
        'dev': ['cudatoolkit>=10.1.243'],
        'test': ['pytest'],
    },

    project_urls={
        'Bug Reports': 'https://github.com/belbahrim/twin-causal-net/issues',
        'Source': 'https://github.com/belbahrim/twin-causal-net',
    },
)
