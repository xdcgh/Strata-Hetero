@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" -S "C:\Users\DC\Documents\ChatGPT\Strata-Hetero\tools\vision" -B "E:\Strata-Hetero-data\build\vision-stage-taps-cpu-20261009-39" -G Ninja -DCMAKE_MAKE_PROGRAM="E:\Strata-Hetero-data\venv\Scripts\ninja.exe" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLAMA_DIR="E:\Strata-Hetero-data\source\llama-3cf0325" -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
set "PHASE_RC=%ERRORLEVEL%"
>"C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-39-vision-stage-cpu-build\configure.exit.txt" echo %PHASE_RC%
exit /b %PHASE_RC%
