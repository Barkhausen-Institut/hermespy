:: This script builds and packages the Windows release of HermesPy.
:: It requires the proper virtual python environment to be activated.

@echo off

:: Make sure the most recent build tools are installed
python -m pip install --upgrade build

:: Query the number of a vailable CPU cores for parallel build and update the MAKEFLAGS environment variable
for /F "delims=" %%A in ('wmic cpu get NumberOfCores /format:value ^| find "NumberOfCores"') do set %%A
set MAKEFLAGS="-j %NumberOfCores%"

:: Build the package from the hermes root directory
pushd  %~dp0\..\..
python -m build --wheel .
popd