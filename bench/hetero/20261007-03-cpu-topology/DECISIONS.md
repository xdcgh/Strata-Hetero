# CPU topology preparation

- Collect Windows CPU Set identifiers, group/core/NUMA identity, efficiency class, allocation and known process-mask restrictions without modifying affinity. Treat CPU Set records as size-delimited and validate truncated/extended payloads.
- The reviewed Windows snapshot reports 16 allowed logical processors in group 0 and numeric efficiency classes 0/1. Exact P/E/LP hardware labels have not been proved. Worker suggestions use the existing pool's class-order rule and remain unmeasured candidates.
- Main review corrected the SDK SchedulingClass union member to a byte and fixed preservation of Linux logical CPU zero. Eleven fixture tests pass, including group identity, mask/default-set restrictions, malformed records, no-observation validation and output preservation.
- No CPU mathematics/performance sweep or affinity change has run. Actual role separation, NUMA/memory constraints, worker sweeps and the 5% acceptance gate remain pending.
- At the quota-resume checkpoint no compiler or Strata GPU process remains, but the other chat is actively preparing IQ4_XS quality inference. Both goals need exclusive GPU capacity. The user requested stopping when a decision is needed; preserve this checkpoint and request GPU priority rather than racing model launches.
