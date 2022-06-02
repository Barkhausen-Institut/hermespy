# from skbuild import setup
from setuptools import find_namespace_packages, setup
from sphinx.setup_command import BuildDoc

cmdclass = {'build_sphinx': BuildDoc}


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hermespy",
    version="0.2.7",
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
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_namespace_packages(include=['hermespy.*'], exclude=['tests']),
    namespace_packages=['hermespy'],
    package_dir={"hermespy": "hermespy"},
    package_data={
      'hermespy.core': ['styles/*.mplstyle']
    },
    include_package_data=True,
    exclude_package_data={
        '': ['3rdparty', 'tests'],
    },
    extras_require={"test": ["pytest"],
                    "quadriga": ["oct2py"]},
    zip_safe=False,
    python_requires=">=3.9",
    entry_points={
        'console_scripts': ['hermes=hermespy.bin:hermes'],
    },
    install_requires=[
        'matplotlib~=3.5.1',
        'numpy~=1.21.5',
        'scipy~=1.7.1',
        'pytest-mypy~=0.9.1',
        'pytest-flake8~=1.1.1',
        'pybind11~=2.6.2',
        'ray~=1.12.0',
        'ruamel.yaml~=0.17.17',
        'sparse~=0.13.0',
        'numba~=0.55.1',
        'sphinx~=4.3.1',
        'rich~=11.2.0',
        'protobuf~=3.19.1',
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', 'HermesPy'),
            'version': ('setup.py', '0.2.7'),
            # 'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docssource'),
            'build_dir': ('setup.py', 'documentation'),
        }
    },
)
