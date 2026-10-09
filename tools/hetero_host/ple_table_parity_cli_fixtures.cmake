if(NOT DEFINED PROGRAM OR NOT DEFINED FIXTURE_DIR)
  message(FATAL_ERROR "PROGRAM and FIXTURE_DIR are required")
endif()
file(MAKE_DIRECTORY "${FIXTURE_DIR}")

execute_process(COMMAND "${PROGRAM}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
string(FIND "${out}" "file_opened" metadata_marker)
if(NOT rc EQUAL 0 OR NOT out MATCHES "metadata_only" OR metadata_marker EQUAL -1 OR NOT out MATCHES "false")
  message(FATAL_ERROR "metadata-only invocation failed: rc=${rc}; out=${out}; err=${err}")
endif()

execute_process(COMMAND "${PROGRAM}" --unknown RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 2)
  message(FATAL_ERROR "unknown argument should return 2, got ${rc}")
endif()

execute_process(COMMAND "${PROGRAM}" --run --file RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 2)
  message(FATAL_ERROR "missing option value should return 2, got ${rc}")
endif()

execute_process(COMMAND "${PROGRAM}" --run RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 2)
  message(FATAL_ERROR "missing required paths should return 2, got ${rc}")
endif()

set(input "${FIXTURE_DIR}/tiny-input.gguf")
set(existing_output "${FIXTURE_DIR}/existing-output.json")
file(WRITE "${input}" "fixture only; this must never be parsed as GGUF\n")
file(WRITE "${existing_output}" "preserve-this-sentinel\n")

execute_process(COMMAND "${PROGRAM}" --run --file "${input}" --file "${input}" --output "${FIXTURE_DIR}/never.json"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 2)
  message(FATAL_ERROR "duplicate --file should return 2, got ${rc}")
endif()

execute_process(COMMAND "${PROGRAM}" --run --file "${input}" --output "${existing_output}" --output "${FIXTURE_DIR}/never2.json"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 2)
  message(FATAL_ERROR "duplicate --output should return 2, got ${rc}")
endif()

execute_process(COMMAND "${PROGRAM}" --run --file "${input}" --output "${existing_output}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 1)
  message(FATAL_ERROR "existing output should be refused before input inspection, got ${rc}")
endif()
file(READ "${existing_output}" sentinel)
if(NOT sentinel STREQUAL "preserve-this-sentinel\n")
  message(FATAL_ERROR "existing output was modified")
endif()
