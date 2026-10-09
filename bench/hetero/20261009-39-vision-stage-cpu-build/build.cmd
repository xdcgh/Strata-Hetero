@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" --build "E:\Strata-Hetero-data\build\vision-stage-taps-cpu-20261009-39" --target strata-vision --parallel 1
set "PHASE_RC=%ERRORLEVEL%"
>"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-39-vision-stage-cpu-build\build.exit.txt" echo %PHASE_RC%
exit /b %PHASE_RC%
