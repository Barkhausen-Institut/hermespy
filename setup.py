from skbuild import setup
from setuptools import find_packages

setup(
    name="hermespy",
    version="0.1.0",
    author="Tobias Kronauer",
    author_email="tobias.kronauer@bi-dd.de",
    description="",
    long_description="",
    packages=find_packages('.', exclude=("tests",)),
    package_dir={"": ""},
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    zip_safe=False
)