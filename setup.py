from skbuild import setup
from setuptools import find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hermespy",
    version="0.2.2",
    author="Jan Adler",
    author_email="jan.adler@barkhauseninstitut.org",
    description="The Heterogeneous Mobile Radio Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/barkhauseninstitut/wicon/hermespy",
    project_urls={
        "Barkhausen Institute": "https://www.barkhauseninstitut.org",
        "Bug Tracker": "https://gitlab.com/barkhauseninstitut/wicon/hermespy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={"": ""},
    include_package_data=True,
    exclude_package_data={
        '': ['3rdparty', 'tests'],
    },
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    python_requires=">=3.9",
    entry_points={
        'console_scripts': ['hermes=hermespy.bin:hermes'],
    },
    install_requires=['matplotlib', 'numpy', 'scipy', 'data-science-types', 'ruamel.yaml', 'numba', 'sparse'],
)
