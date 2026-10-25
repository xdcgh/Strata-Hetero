@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > "E:\Strata-Hetero-data\build\cpu-q8-boundary74-01\vcvars.log" 2>&1
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /MT /O2 /I"E:\Strata-Hetero-data\source\llama-3cf0325\ggml\include" /I"E:\Strata-Hetero-data\source\llama-3cf0325\ggml\src" "C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261025-74-verify-finite\preparation\cpu_q8_1_boundary.cpp" /Fo"E:\Strata-Hetero-data\build\cpu-q8-boundary74-01\boundary.obj" /Fe"E:\Strata-Hetero-data\build\cpu-q8-boundary74-01\boundary.exe" "E:\Strata-Hetero-data\build\cuda-verify-probe74-02\ggml\src\ggml-cpu.lib" "E:\Strata-Hetero-data\build\cuda-verify-probe74-02\ggml\src\ggml-base.lib" > "E:\Strata-Hetero-data\build\cpu-q8-boundary74-01\build.log" 2>&1
exit /b %errorlevel%
