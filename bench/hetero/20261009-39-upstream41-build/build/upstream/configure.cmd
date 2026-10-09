@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > "E:\Strata-Hetero-data\build\cuda-upstream41-03\configure.vcvars.log" 2>&1
if errorlevel 1 goto vcvars_failed
set "CUDA_PATH=E:\Strata-Hetero-data\toolchains\cuda-13.3.1"
set "CUDA_PATH_V13_3=E:\Strata-Hetero-data\toolchains\cuda-13.3.1"
set "CUDACXX=E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin\nvcc.exe"
set "PATH=E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin;E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin\x64;E:\Strata-Hetero-data\toolchains\cuda-13.3.1\nvvm\bin;%PATH%"
where cl > "E:\Strata-Hetero-data\build\cuda-upstream41-03\where-cl.log" 2>&1
where nvcc > "E:\Strata-Hetero-data\build\cuda-upstream41-03\where-nvcc.log" 2>&1
cl /Bv > "E:\Strata-Hetero-data\build\cuda-upstream41-03\cl-version.log" 2>&1
"E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin\nvcc.exe" --version > "E:\Strata-Hetero-data\build\cuda-upstream41-03\nvcc-version.log" 2>&1
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" --version > "E:\Strata-Hetero-data\build\cuda-upstream41-03\cmake-version.log" 2>&1
"E:\Strata-Hetero-data\venv\Scripts\ninja.exe" --version > "E:\Strata-Hetero-data\build\cuda-upstream41-03\ninja-version.log" 2>&1
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" "-S" "E:\Strata-Hetero-data\source\upstream-fb58-build-01" "-B" "E:\Strata-Hetero-data\build\cuda-upstream41-03" "-G" "Ninja" "-DCMAKE_MAKE_PROGRAM=E:\Strata-Hetero-data\venv\Scripts\ninja.exe" "-DCMAKE_BUILD_TYPE=Release" "-DSTRATA_ENABLE_CUDA=ON" "-DCMAKE_CUDA_ARCHITECTURES=89" "-DCMAKE_CUDA_COMPILER=E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin\nvcc.exe" "-DCUDAToolkit_ROOT=E:\Strata-Hetero-data\toolchains\cuda-13.3.1" "-DSTRATA_BUILD_TESTS=OFF" "-DSTRATA_NATIVE_EXPERTS=ON" "-DSTRATA_PORTABLE=ON" "-DSTRATA_GGML_DIR=E:\Strata-Hetero-data\source\llama-3cf0325" "-DSTRATA_ENABLE_HIP=OFF" "-DSTRATA_HIP_GFX906=OFF" "-DSTRATA_ENABLE_SYCL=OFF" "-DSTRATA_MMQ_KQUANTS=OFF" "-DSTRATA_Q6K_EXPERTS=OFF" "-DSTRATA_PREFILL_MMQ=OFF" "-DSTRATA_ORCA_Q4KS_MMQ=OFF" "-DSTRATA_ISA_FLOOR=" > "E:\Strata-Hetero-data\build\cuda-upstream41-03\configure.log" 2>&1
set "RC=%ERRORLEVEL%"
>"E:\Strata-Hetero-data\build\cuda-upstream41-03\configure.exit.txt" echo %RC%
exit /b %RC%
:vcvars_failed
set "RC=%ERRORLEVEL%"
>"E:\Strata-Hetero-data\build\cuda-upstream41-03\configure.exit.txt" echo %RC%
exit /b %RC%
