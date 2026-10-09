@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"E:\Strata-Hetero-data\build\vision-stage-fixture-20261009-39\vision-stage-trace-fixture.exe" "E:\Strata-Hetero-data\vision-fixtures\stage-metadata-20261009-39"
set "PHASE_RC=%ERRORLEVEL%"
>"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-39-vision-stage-cpu-build\fixture-execute.exit.txt" echo %PHASE_RC%
exit /b %PHASE_RC%
