:: This script builds and packages the Windows release of HermesPy.
:: It requires the proper virtual python environment to be activated.

@echo off

:: Make sure the most recent build tools are installed
python -m pip install --upgrade setuptools

:: Remove possibly deprecated build artifacts
IF EXIST %~dp0\..\..\hermespy.egg-info DEL /F /Q %~dp0\..\..\hermespy.egg-info

:: Build the package from the hermes root directory
pushd  %~dp0\..\..
python -m build --sdist .
popd