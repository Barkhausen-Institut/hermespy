from skbuild import setup
from setuptools import find_packages

setup(
    name="hermespy",
    author="Tobias Kronauer",
    author_email="tobias.kronauer@bi-dd.de",
    description="",
    long_description="",
    packages=find_packages('.', exclude=("tests",)),
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    zip_safe=False
)