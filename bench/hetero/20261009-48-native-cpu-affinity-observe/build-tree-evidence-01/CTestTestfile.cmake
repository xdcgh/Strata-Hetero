# CMake generated Testfile for 
# Source directory: C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu
# Build directory: E:/Strata-Hetero-data/build/native-cpu-affinity-observe-20261009-48
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("hetero_worker_affinity_observation" "E:/Strata-Hetero-data/build/native-cpu-affinity-observe-20261009-48/hetero_worker_affinity_observation.exe")
set_tests_properties("hetero_worker_affinity_observation" PROPERTIES  TIMEOUT "15" _BACKTRACE_TRIPLES "C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu/CMakeLists.txt;90;add_test;C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu/CMakeLists.txt;0;")
add_test("hetero_native_expert_metadata_fixture" "E:/Strata-Hetero-data/build/native-cpu-affinity-observe-20261009-48/hetero_native_expert_metadata_fixture.exe")
set_tests_properties("hetero_native_expert_metadata_fixture" PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu/CMakeLists.txt;92;add_test;C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu/CMakeLists.txt;0;")
add_test("hetero_native_expert_cli_fixtures" "E:/Strata-Hetero-data/venv/Lib/site-packages/cmake/data/bin/cmake.exe" "-DEXECUTABLE=E:/Strata-Hetero-data/build/native-cpu-affinity-observe-20261009-48/hetero_native_expert.exe" "-DFIXTURE_DIR=E:/Strata-Hetero-data/build/native-cpu-affinity-observe-20261009-48/cli-fixtures" "-P" "C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu/cli_fixtures.cmake")
set_tests_properties("hetero_native_expert_cli_fixtures" PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu/CMakeLists.txt;93;add_test;C:/Users/DC/Documents/ChatGPT/Strata-Hetero/tools/hetero_native_cpu/CMakeLists.txt;0;")
subdirs("strata-root")
