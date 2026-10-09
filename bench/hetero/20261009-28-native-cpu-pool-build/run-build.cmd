@echo off
setlocal
set /a identity_wait=0
:wait_for_process_identity
if exist "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\process-identity-verified.txt" goto process_identity_verified
set /a identity_wait+=1
if %identity_wait% GEQ 60 exit /b 92
timeout /t 1 /nobreak >nul
goto wait_for_process_identity
:process_identity_verified
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\vcvars.log" 2>&1
if errorlevel 1 (
  >"E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\configure.exit.txt" echo 1
  exit /b 1
)
set "CUDA_PATH="
set "CUDA_HOME="
set "CUDACXX="
set "CUDA_VISIBLE_DEVICES="
set "HIP_PATH="
set "ROCM_PATH="
set "ONEAPI_ROOT="
set "ONEAPI_DEVICE_SELECTOR="
set "SYCL_DEVICE_FILTER="
where cl > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\tool-paths.log" 2>&1
where cmake >> "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\tool-paths.log" 2>&1
where ninja >> "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\tool-paths.log" 2>&1
cl /Bv > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\cl-version.log" 2>&1
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" --version > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\cmake-version.log" 2>&1
"E:\Strata-Hetero-data\venv\Scripts\ninja.exe" --version > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\ninja-version.log" 2>&1
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" -S "C:\Users\DC\Documents\ChatGPT\Strata-Hetero\tools\hetero_native_cpu" -B "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28" -G Ninja "-DCMAKE_MAKE_PROGRAM=E:\Strata-Hetero-data\venv\Scripts\ninja.exe" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF -DSTRATA_ENABLE_SYCL=OFF -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_ORCA_Q4KS_MMQ=OFF -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON "-DSTRATA_ISA_FLOOR=" "-DSTRATA_GGML_DIR=E:\Strata-Hetero-data\source\llama-3cf0325" "-DHETERO_NATIVE_GGML_DIR=E:\Strata-Hetero-data\source\llama-3cf0325" > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\configure.log" 2>&1
set "configure_rc=%errorlevel%"
>"E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\configure.exit.txt" echo %configure_rc%
if not "%configure_rc%"=="0" exit /b %configure_rc%
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" --build "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28" --target hetero_native_expert hetero_native_expert_metadata_fixture --parallel 1 > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\build.log" 2>&1
set "build_rc=%errorlevel%"
>"E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\build.exit.txt" echo %build_rc%
if not "%build_rc%"=="0" exit /b %build_rc%
"E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe" -E env --unset=CUDA_VISIBLE_DEVICES --unset=ONEAPI_DEVICE_SELECTOR --unset=SYCL_DEVICE_FILTER "E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\ctest.exe" --test-dir "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28" --output-on-failure --parallel 1 -R "^hetero_native_expert_" > "E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\ctest.log" 2>&1
set "ctest_rc=%errorlevel%"
>"E:\Strata-Hetero-data\build\native-cpu-pool-20261009-28\ctest.exit.txt" echo %ctest_rc%
exit /b %ctest_rc%
