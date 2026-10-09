@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > "E:\Strata-Hetero-data\build\cuda-hetero41-01\build.vcvars.log" 2>&1
if errorlevel 1 goto vcvars_failed
set "CUDA_PATH=E:\Strata-Hetero-data\toolchains\cuda-13.3.1"
set "CUDA_PATH_V13_3=E:\Strata-Hetero-data\toolchains\cuda-13.3.1"
set "CUDACXX=E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin\nvcc.exe"
set "PATH=E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin;E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin\x64;E:\Strata-Hetero-data\toolchains\cuda-13.3.1\nvvm\bin;%PATH%"
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" "--build" "E:\Strata-Hetero-data\build\cuda-hetero41-01" "--target" "strata" "--parallel" "1" "--verbose" > "E:\Strata-Hetero-data\build\cuda-hetero41-01\build.log" 2>&1
set "RC=%ERRORLEVEL%"
>"E:\Strata-Hetero-data\build\cuda-hetero41-01\build.exit.txt" echo %RC%
exit /b %RC%
:vcvars_failed
set "RC=%ERRORLEVEL%"
>"E:\Strata-Hetero-data\build\cuda-hetero41-01\build.exit.txt" echo %RC%
exit /b %RC%
