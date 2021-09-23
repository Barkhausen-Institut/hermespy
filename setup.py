from skbuild import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hermespy",
    version="0.1.0",
    author="Tobias Kronauer",
    author_email="tobias.kronauer@barkhauseninstitut.org",
    description="The Heterogeneous Mobile Radio Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/barkhauseninstitut/wicon/hermespy",
    project_urls={
        "Barkhausen Institute": "https://www.barkhauseninstitut.org",
        "Bug Tracker": "https://gitlab.com/barkhauseninstitut/wicon/hermespy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License version 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages('.', exclude=("tests",)),
    package_dir={"": ""},
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    python_requires=">=3.7",
)
