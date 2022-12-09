from skbuild import setup
from setuptools import find_namespace_packages
from sphinx.setup_command import BuildDoc

cmdclass = {'build_sphinx': BuildDoc}


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hermespy",
    version="1.0.0",
    author="Jan Adler",
    author_email="jan.adler@barkhauseninstitut.org",
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
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_namespace_packages(include=['hermespy.*']),
    package_data={
        'hermespy.core': ['styles/*.mplstyle'],
        'hermespy.channel': ['res/*'],
    },
    include_package_data=True,
    exclude_package_data={
        '': ['3rdparty', 'tests'],
    },
    extras_require={
        "test": ['pytest', 'coverage'],
        "quadriga": ["oct2py"],
        "documentation": [
            'sphinx-autodoc-typehints',
            'sphinxcontrib-apidoc',
            'sphinxcontrib-mermaid',
            'sphinxcontrib-bibtex',
            'sphinx-rtd-theme',
            'sphinx-rtd-dark-mode',
            'sphinx-tabs',
            'sphinx-copybutton',
            'nbsphinx',
            'ipywidgets',
            'scikit-build',
        ],
        "uhd": ['usrp_uhd_api@git+https://github.com/Barkhausen-Institut/usrp_uhd_wrapper.git'],
        "audio": ['sounddevice'],
        "develop": [
            "pybind11",
            "scikit-build",
            "cmake",
            "sphinx",
            "wheel",
            "black",
            "flake8",
        ]
    },
    zip_safe=False,
    python_requires=">=3.9",
    entry_points={
        'console_scripts': ['hermes=hermespy.bin:hermes'],
    },
    install_requires=[
        'matplotlib~=3.6.2',
        'h5py~=3.7.0',
        'numpy>=1.21.6',
        'scipy~=1.9.3',
        'pybind11~=2.10.1',
        'ray~=2.1.0',
        'ruamel.yaml~=0.17.21',
        'sparse~=0.13.0',
        'numba~=0.56.4',
        'sphinx~=5.3.0',
        'rich~=12.6.0',
        'ZODB~=5.7.0',
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', 'HermesPy'),
            'version': ('setup.py', '1.0.0'),
            # 'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docssource'),
            'build_dir': ('setup.py', 'documentation'),
        }
    },
)
