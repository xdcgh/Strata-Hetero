# Latest integrated full-table RAM quality failure

The full 28,800,138,240-byte IQ4_NL table was locked and the server reached ready with a 20 GiB pinned expert complement. Engine binary/source, model, tokenizer, expert/GPU placement and request controls match the accepted direct control except the declared PLE mode/reserve. No inherited STRATA experiment variables were present to clear, and no debug synchronization flag was enabled.

Four short tasks passed all 12 formal token/text comparisons. The three-key retrieval at prompt usage 114/cache_n=0 failed all three formals. The first raw-ID difference is zero-based index 65: direct ID 19 versus RAM ID 0. Each RAM response then ends with 63 ID-0 tokens and 63 visible exclamation marks. The output cap remains 128 and transport was normal. This is degeneration after a correct prefix, not harmless cap exhaustion. Four longer tasks were stopped.

Earlier generic capped-response reports were preserved. The root added a tail-loop detector, passed nine quality and 24 comparator fixtures, and saved an additional reviewed report. The complete comparator observes only 15 of 27 required pairs and does not accept the run.

`root-resource-window-review.json` distinguishes exact client request intervals from the whole sampler lifetime. The fifth task's warmup/formal windows contain 26 samples, with minima 47.578 GiB available physical RAM and 18.427 GiB available system commit and no alerts/errors. Whole-lifetime minima are 38.205/18.427 GiB across 1,475 samples. The prior diagnostic used the whole-stream count in a block named for the fifth task; its original bytes remain. These observations do not establish a memory-exhaustion cause.

The exact owned model, bridge and sampler were unloaded/stopped and port 8081 released. Full-table mode remains experimental and is not accepted for daily use. Next isolate the row/gather/dequant paths without a GPU, then use targeted runtime evidence to distinguish mapped-data issues from synchronization. Do not conceal the failure with repeated restarts or infer that an instrumented success proves a fix.
