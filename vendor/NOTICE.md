# Upstream source and modifications

Source: https://github.com/zhentingqi/EoM

Revision: `d272266f7d6a0c302b7af5d20f1326b63c5656fb` (the upstream HEAD fetched
when this adapter was implemented).

`SOURCE_MANIFEST.json` lists the SHA-256 of every copied source file at that
revision. The test suite checks all entries marked unmodified. Original project
metadata and documentation are preserved as `EoM.pyproject.toml` and `EoM.README.md`.
Upstream project metadata declares MIT; that revision does not contain a separate
top-level license file. No license from its unrelated third-party dependencies
has been substituted. Original source attribution is retained.

Copied source includes the complete base agent/environment/population/engine/
pipeline/configuration implementation; the researchworld agent, environment and
prompts; and the logging/LLM utilities needed by those imports. Large benchmark
datasets, runs, CloudCast third-party execution dependencies and hardware
simulator workspaces are not copied.

Only three copied files were changed:

- `hayekmas/__init__.py`: lazy exports instead of importing every original runtime.
- `hayekmas/base/__init__.py`: lazy engine/pipeline imports; base interfaces remain direct exports.
- `hayekmas/adapters/researchworld/__init__.py`: exports only the bundled adapter primitives.

All cooperative mechanisms, metering, tests, examples and replay code are new
under `ant_colony/`, `tests/`, `configs/` and `examples/`. Original bid selection,
individual wealth updates, and path-reward functions are not called by the new
engine. Keeping those files unmodified makes the departure from EoM inspectable.

Provider protocol references checked during implementation:

- https://openrouter.ai/docs/guides/routing/provider-selection#max-price
- https://openrouter.ai/docs/api_reference/overview

Provider model names and prices are intentionally supplied by the experiment
operator, not pinned to potentially stale examples in upstream documentation.

The ProgramBench extension additionally reuses local `code-testing-v1` helpers
for reproducible reference probes, the pinned native summary API, and durable
campaign reservations. `PROGRAMBENCH_ADAPTER_SOURCE.json` records their exact
source hashes and adaptations. The benchmark implementation and data remain
external at https://github.com/facebookresearch/ProgramBench, pinned to
`27f02157c785f8da3647aa6dbbe6b9137f99f10e`. No benchmark solution or hidden test
is copied into agent inputs or the colony implementation.
