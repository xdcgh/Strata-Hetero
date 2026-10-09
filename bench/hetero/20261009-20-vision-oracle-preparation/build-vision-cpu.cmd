@echo off
setlocal
set "ROOT=C:\Users\DC\Documents\ChatGPT\Strata-Hetero"
set "LLAMA=E:\Strata-Hetero-data\source\llama-3cf0325"
set "BUILD=E:\Strata-Hetero-data\build\vision-oracle-cpu-20261009-20"
set "CMAKE=E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe"
set "NINJA=E:\Strata-Hetero-data\venv\Scripts\ninja.exe"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" -S "%ROOT%\tools\vision" -B "%BUILD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="%LLAMA%" -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%BUILD%" --target strata-vision --parallel 1
exit /b %errorlevel%
