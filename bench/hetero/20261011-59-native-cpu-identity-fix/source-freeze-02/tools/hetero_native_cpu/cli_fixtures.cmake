if(NOT DEFINED EXECUTABLE OR NOT DEFINED FIXTURE_DIR)
  message(FATAL_ERROR "EXECUTABLE and FIXTURE_DIR are required")
endif()

file(MAKE_DIRECTORY "${FIXTURE_DIR}")
foreach(name IN ITEMS denied.bin protected.bin failure.json wrong-size.bin bad-size.json bad-size-output.bin invalid-type.bin invalid-type.json)
  if(EXISTS "${FIXTURE_DIR}/${name}")
    message(FATAL_ERROR "refusing to overwrite pre-existing CLI fixture path: ${FIXTURE_DIR}/${name}")
  endif()
endforeach()

function(expect_success label)
  execute_process(COMMAND "${EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${FIXTURE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "${label} failed (${result}): ${stderr}${stdout}")
  endif()
endfunction()

function(expect_failure label)
  execute_process(COMMAND "${EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${FIXTURE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if("${result}" STREQUAL "0")
    message(FATAL_ERROR "${label} unexpectedly succeeded: ${stdout}")
  endif()
  if(NOT stderr MATCHES "hetero_native_expert:")
    message(FATAL_ERROR "${label} exited without a captured CLI diagnostic (${result}): ${stderr}${stdout}")
  endif()
endfunction()

# Metadata mode exercises the explicit and default parser paths without model files or output writes.
expect_success("default metadata")
expect_success("explicit metadata" --validate-only --rows 1 --hidden 2560 --intermediate 640 --gu-type 12 --down-type 7)
# Pool options are parser/metadata-only here: they must not construct a pool or touch model data.
foreach(affinity IN ITEMS none all auto p-cores)
  expect_success("pool metadata ${affinity}" --validate-only --rows 1 --hidden 2560 --intermediate 640
    --gu-type 12 --down-type 7 --pool-workers 1 --pool-affinity ${affinity})
endforeach()
expect_success("pool worker lower bound" --validate-only --pool-workers 1)
expect_success("pool worker upper bound" --validate-only --pool-workers 64)
expect_failure("unknown option" --validate-only --not-a-real-option)
expect_failure("bounded rows" --validate-only --rows 257)
expect_failure("unsupported native type" --validate-only --gu-type 999)
expect_failure("pool worker below bound" --validate-only --pool-workers 0)
expect_failure("pool worker above bound" --validate-only --pool-workers 65)
expect_failure("pool affinity requires workers" --validate-only --pool-affinity auto)
expect_failure("unsupported pool affinity" --validate-only --pool-workers 1 --pool-affinity random)
expect_failure("metadata output option requires run" --validate-only --output "${FIXTURE_DIR}/denied.bin")
if(EXISTS "${FIXTURE_DIR}/denied.bin")
  message(FATAL_ERROR "metadata mode created an output file")
endif()
file(GLOB metadata_outputs "${FIXTURE_DIR}/*")
if(metadata_outputs)
  message(FATAL_ERROR "metadata-only CLI created files: ${metadata_outputs}")
endif()

# An existing output is never replaced. The safe, new receipt path captures this failure.
file(WRITE "${FIXTURE_DIR}/protected.bin" "sentinel-output\n")
expect_failure("no-overwrite run preflight" --run --blob missing-blob.bin --input missing-input.f32 --rows 1
  --hidden 2560 --intermediate 640 --gu-type 12 --down-type 7
  --output "${FIXTURE_DIR}/protected.bin" --receiptJSON "${FIXTURE_DIR}/failure.json")
file(READ "${FIXTURE_DIR}/protected.bin" protected_contents)
if(NOT protected_contents STREQUAL "sentinel-output\n")
  message(FATAL_ERROR "pre-existing output was modified")
endif()
file(READ "${FIXTURE_DIR}/failure.json" failure_receipt)
if(NOT failure_receipt MATCHES "status.*failed" OR NOT failure_receipt MATCHES "already exists")
  message(FATAL_ERROR "no-overwrite failure receipt lacks its cause: ${failure_receipt}")
endif()

# Exact byte-size validation rejects an invalid raw blob before any native kernel can run.
file(WRITE "${FIXTURE_DIR}/wrong-size.bin" "x")
expect_failure("unsupported run type" --run --blob missing-blob.bin --input missing-input.f32 --rows 1
  --hidden 2560 --intermediate 640 --gu-type 999 --down-type 7
  --output "${FIXTURE_DIR}/invalid-type.bin" --receiptJSON "${FIXTURE_DIR}/invalid-type.json")
file(READ "${FIXTURE_DIR}/invalid-type.json" invalid_type_receipt)
if(NOT invalid_type_receipt MATCHES "GGML type ID is outside" OR EXISTS "${FIXTURE_DIR}/invalid-type.bin")
  message(FATAL_ERROR "unsupported type did not preserve an error receipt without writing output: ${invalid_type_receipt}")
endif()

expect_failure("bad blob size" --run --blob "${FIXTURE_DIR}/wrong-size.bin" --input missing-input.f32 --rows 1
  --hidden 2560 --intermediate 640 --gu-type 12 --down-type 7
  --output "${FIXTURE_DIR}/bad-size-output.bin" --receiptJSON "${FIXTURE_DIR}/bad-size.json")
file(READ "${FIXTURE_DIR}/bad-size.json" bad_size_receipt)
if(NOT bad_size_receipt MATCHES "size mismatch" OR EXISTS "${FIXTURE_DIR}/bad-size-output.bin")
  message(FATAL_ERROR "blob-size failure receipt lacks size mismatch: ${bad_size_receipt}")
endif()

file(REMOVE "${FIXTURE_DIR}/protected.bin" "${FIXTURE_DIR}/failure.json"
  "${FIXTURE_DIR}/wrong-size.bin" "${FIXTURE_DIR}/bad-size.json" "${FIXTURE_DIR}/invalid-type.json")
