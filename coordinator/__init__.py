"""The VLM coordinator around a frozen VLA executor (ECE 780 proposal, Figure 1). Guide: README.md.

Modules: schema (records), memory (M_t), context (memory -> VLM requests), prompts, loop (the Coordinator), configs,
executor (GR00T), paths. Subpackages: vlm (model backends: cloud APIs and local servers), tracking, detection, recovery,
motion (retrace, backoff). Importing the package loads nothing heavy; the simulator is imported only by the modules
that need it."""
