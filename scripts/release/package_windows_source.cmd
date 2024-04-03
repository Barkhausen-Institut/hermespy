:: This script builds and packages the Windows release of HermesPy.
:: It requires the proper virtual python environment to be activated.

@echo off

:: Make sure the most recent build tools are installed
python -m pip install --upgrade setuptools

:: Remove possibly deprecated build artifacts
IF EXIST %~dp0\..\..\MANIFEST DEL /F %~dp0\..\..\MANIFEST
IF EXIST %~dp0\..\..\MANIFEST.in DEL /F %~dp0\..\..\MANIFEST.in
IF EXIST %~dp0\..\..\_skbuild DEL /F /Q %~dp0\..\..\_skbuild
IF EXIST %~dp0\..\..\hermespy.egg-info DEL /F /Q %~dp0\..\..\hermespy.egg-info
COPY %~dp0\SOURCE_MANIFEST.in %~dp0..\..\MANIFEST.in

:: Build the package from the hermes root directory
pushd  %~dp0\..\..
python setup.py --skip-cmake sdist --formats=gztar
popd