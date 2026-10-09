@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe" /nologo /std:c++17 /EHsc /O2 /MT /arch:AVX2 /I"E:\Strata-Hetero-data\source\llama-3cf0325\ggml\include" /Fo"E:\Strata-Hetero-data\build\vision-stage-fixture-20261009-39\fixture.obj" /Fe"E:\Strata-Hetero-data\build\vision-stage-fixture-20261009-39\vision-stage-trace-fixture.exe" "C:\Users\DC\Documents\ChatGPT\Strata-Hetero\tests\core\vision_stage_trace_test.cpp"
set "PHASE_RC=%ERRORLEVEL%"
>"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-39-vision-stage-cpu-build\fixture-compile.exit.txt" echo %PHASE_RC%
exit /b %PHASE_RC%
