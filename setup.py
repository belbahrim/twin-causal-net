#Using setuptools for installation
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for uploading to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of the project. 
    # It will determine how users can install this project, e.g.:
    #
    # $ pip install twincausal
    #
    # And where it will live on PyPI: https://pypi.org/project/twincausal/
    #

    name='twincausal',  # Required

    # Versions comply with PEP 440:
    # twincausal version convention, ("major.minor.micro")
    #
    version='0.1.6',  # Required

    # This is a one-line description or tagline of what the project does. This
    # corresponds to the "Summary" metadata field:
    description='Twin Neural Network based causal models for Uplift estimation',  # Optional
    license='MIT License',
    # This is an optional longer description of the project that represents
    # the body of text which users will see when users visit PyPI.
    #
    # This is the same as the README, so reading it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    long_description=long_description,  # Optional

    # Denotes that our long_description is in Markdown; other supported valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    long_description_content_type='text/markdown',  # Optional (see note above)

    # This should a valid link to the project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    url='https://code.td.com/projects/ADV_PROJ/repos/twin-causal-model/',  # Optional

    author='Belbahri, M., Gandouet, O., and Sharoff. P',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email=['mouloud@td.com', 'oliver@td.com', 'sharoff.ponkumar@tdinsurance.com'],  # Optional

    # Classifiers help users find the project by categorizing it.
    classifiers=[  # Optional
        # values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who the project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # License we currently use
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions that are supported here. In particular, ensure
        # that the project support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    # This field adds keywords for the project which will appear on the
    # project page. 
    #
    # Note that this is a list of additional keywords, separated
    # by commas, to be used to assist searching for the distribution in a
    # larger catalog.
    keywords='causal_inference, deeplearning,machinelearning',  # Optional

    # When the source code is in a subdirectory under the project root, 
    # it is necessary to specify the `package_dir` argument.
    package_dir={'': 'src'},  # Optional

    #
    # Alternatively, if distribution can be done for a single Python file, for that use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(where='src'),  # Required

    # Specify which Python versions the project support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. 
    python_requires='>=3.6, <4',

    # This field lists other packages that this project depends on to run.
    # Any package in the list will be installed by pip when the project is
    # installed, so they must be valid existing projects.
    #
    install_requires=['numpy>=1.19.2',
    'scikit-learn>=0.23.2',
    'pandas>=1.1.3',
    'matplotlib>=3.3.2',
    'scipy>=1.5',
    'tensorboard>=2.3.0',
    'tensorboardX>=1.9',
    'tensorboard-plugin-wit>=1.7.0',
    # 'yaml>=0.2.5'
    ],  # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install twincausal[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        'dev': ['cudatoolkit>=10.1.243'],
        'test': ['pytest'],
    },


    # List additional URLs that are relevant to the project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        # 'Bug Reports': 'https://github.com/pypa/sampleproject/issues',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://code.td.com/projects/ADV_PROJ/repos/twin-causal-model/',
    },
)