# Actual PLE host-path parity

The tested CPU-only executable reuses the engine's PleTable and DirectFile code. Its four host fixtures passed before the explicit real-file run. The first compile failure and monitor-wrapper error remain in build-evidence; the corrected compile/source hashes and successful fixtures are separate.

Actual Direct, pageable mmap and fully locked mmap outputs all agree bit for bit across five batch sizes, 15 warmups and 45 formal gathers. Every output has 1,310,720 finite floats. Each arm's 8,192 read_row checks agree with its batch gather, including row zero, a page-straddling row and the final row. The whole generated row-ID buffer hash and standard-library generator/endianness are recorded. The three arms opened and closed sequentially, so a Direct read did not overlap a live mmap in this process.

The complete table lock is exactly 28,800,138,240 bytes. The one lock startup observation is about 110 seconds and is not a repeated cold/warm startup measurement. All source identities and observed memory floors passed; only this owned CPU executable ran, and its sampler/target exited.

This narrows the RAM model investigation by verifying the sampled host data paths. It does not prove every row, GPU staging, MTP verification or model computation correct. Preserve the latest model failure, add targeted evidence around its divergence, and keep full-table mode unaccepted until an uninstrumented model gate passes.
