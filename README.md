# Swarm of Ants

A research prototype for collaborative agents that share a bank, form teams, and delegate work recursively. Agents receive bounded local context and can propose work, join teams, hire or spawn collaborators, communicate, and sleep. The scheduler enforces budgets, communication and hierarchy limits, and round boundaries; it does not prescribe fixed worker roles.

The project is derived from [Economy of Minds](https://github.com/zhentingqi/EoM). See [`vendor/NOTICE.md`](vendor/NOTICE.md) for the pinned source revision and provenance. The mechanism and its scalability assumptions are described in [`DESIGN.md`](DESIGN.md).

## Install and test

Requires Python 3.11 or newer.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

The current local test suite passes **254 tests**. See [`VALIDATION.md`](VALIDATION.md) for the validation scope and limitations.

Run the deterministic demo with:

```sh
ant-colony demo --config configs/demo.json --out runs/demo
```

Experiment scripts and example configurations are under `scripts/` and `configs/`. Live model runs require operator-supplied provider credentials and task data; no API keys or experiment run records are included here.

## Repository contents

- `ant_colony/`: cooperative engine, funding, delegation, bounded context, coding adapter, ProgramBench integration, and replay UI.
- `hayekmas/` and `vendor/`: attributed upstream implementation and source manifests.
- `tests/`: deterministic mechanism, accounting, context, coding, and benchmark-adapter tests.
- `examples/` and `configs/`: public fixtures and example settings.

Generated run logs, budget ledgers, reports, virtual environments, and media are intentionally excluded from the repository.
