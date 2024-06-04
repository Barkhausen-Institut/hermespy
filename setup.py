from skbuild import setup
from setuptools import find_namespace_packages

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hermespy",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="The Heterogeneous Radio Mobile Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Barkhausen-Institut/hermespy",
    project_urls={
        "Documentation": "https://hermespy.org/",
        "Barkhausen Institute": "https://www.barkhauseninstitut.org",
        "Bug Tracker": "https://github.com/Barkhausen-Institut/hermespy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    cmake_with_sdist=True,      # Do not remove, required for sdist to copy all submodule files into the package
    packages=find_namespace_packages(include=['hermespy.*']),
    package_data={
        'hermespy.core': ['styles/*.mplstyle'],
        'hermespy.channel': ['quadriga/launch_quadriga.m'],
    },
    include_package_data=True,
    extras_require={
        "test": [
            'pytest>=7.4.2',
            'coverage>=7.5.1',
            'mypy>=1.3.0,<1.7.0',
            'nbformat',
            'nbconvert'
        ],
        "quadriga": ["oct2py>=5.6.0"],
        "documentation": [
            'sphinx>=7.0.1',
            'furo>=2023.9.10',
            'sphinx-autodoc-typehints>=1.23.0',
            'sphinxcontrib-apidoc>=0.4.0',
            'sphinxcontrib-mermaid>=0.9.2',
            'sphinxcontrib-bibtex>=2.5.0',
            'sphinx-tabs>=3.4.1',
            'sphinx-copybutton>=0.5.2',
            'sphinx-carousel>=1.2.0',
            'nbsphinx>=0.9.2',
            'ipywidgets>=8.1.1'
        ],
        "uhd": ['usrp-uhd-client>=1.5.0'],
        "audio": ['sounddevice>=0.4.6'],
        "develop": [
            'pybind11>=2.10.4',
            "scikit-build>=0.17.6",
            "cmake>=3.27.2",
            "wheel>=0.41.2",
            "black>=24.4.2",
            "flake8>=6.0.0",
            'coverage>=7.2.7',
            'mypy>=1.3.0',
        ]
    },
    zip_safe=False,
    python_requires=">=3.9",
    entry_points={
        'console_scripts': ['hermes=hermespy.bin:hermes_simulation'],
    },
    install_requires=[
        "numpy>=1.25.2",
        "matplotlib>=3.8.0",
        'h5py>=3.8.0',
        'scipy>=1.1.3',
        'pybind11>=2.10.4',
        'ray>=2.7.0',
        'ruamel.yaml>=0.17.32',
        'sparse>=0.14.0',
        'numba>=0.58.0',
        'nptyping>=2.5.0',
        'rich>=13.6.0',
        'ZODB~=5.8.0',
    ],
)
