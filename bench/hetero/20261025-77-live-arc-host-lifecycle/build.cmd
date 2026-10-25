@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > vcvars.log 2>&1
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++20 /EHsc /MT /O2 /I"E:\Strata-Hetero-data\source\hetero-live-arc-20261025\include" "E:\Strata-Hetero-data\source\hetero-live-arc-20261025\src\kernels\cpu\live_helper.cpp" "E:\Strata-Hetero-data\source\hetero-live-arc-20261025\src\kernels\cpu\live_helper_host_fixture.cpp" /Fe"E:\Strata-Hetero-data\build\live-arc-host77-01\live_helper_host_fixture.exe" > build.log 2>&1
exit /b %errorlevel%
