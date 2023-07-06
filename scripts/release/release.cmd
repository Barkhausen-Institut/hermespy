:: This script uploads distributions to the Python Package Index (PyPI).
:: It requires the proper virtual python environment to be activated.

@echo off

:: Decide which repository to upload to
SET repository=
SET arg1=%1
IF "%arg1%" == "test" SET repository=testpypi
IF "%arg1%" == "production" SET repository=pypi
IF "%repository%" == "" (
    echo Please specify a repository to upload to, either test or production
    exit /b 1
)

:: Make sure the most recent build tools are installed
python -m pip install --upgrade twine

:: Upload the package from the hermes root directory
pushd  %~dp0\..\..
python -m twine upload --repository %repository% dist/*
popd