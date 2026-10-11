# Mechanism and research questions

The experiment asks whether shared capital and local delegation can improve task
quality per token as population grows. Giving every model a shared-wealth prompt
alone would not answer it: the accounting, information access and opportunity to
stop spending must implement that objective.

## What changes from EoM

| Original EoM behavior | Cooperative adapter |
|---|---|
| Wakeup can invoke a model | Timers and message/allocation events wake agents |
| Active agents bid; highest bid buys the next action | Bounded proposals receive bank grants; funded teams work |
| Per-agent wealth and previous-winner payment | Shared team wallets with recursively delegated rights |
| Path rewards distributed over acting agents | One environment contribution score, one reward to bank |
| Bankruptcy repair / periodic births with endowments | Agent-requested, team-funded births with no endowment |
| Shared environment state can grow with history | Bounded local context, scoped scratchpad, archival logs |
| Terminal-tagged role / terminal auction | Automatic coding deadline or explicit early submission; selection before grading |

The original `HayekMAS`, configuration, population and pipeline source is retained
under `hayekmas/base` for review. `ColonyAgent` subclasses the original `BaseAgent`;
the engine uses the original `Population`; `ExactEnv` subclasses `BaseEnv`; rubric
evaluation invokes the original `ResearchEnv` and `ResearchAction`. There is no
dependency on the sibling team's auction implementation and no path-reward call
in the cooperative engine. Reward magnitudes are credits configured by the
experiment, not real money earned from solving tasks.

Uniform virtual token prices do not fully represent heterogeneous model prices.
Maximizing this wealth objective can reduce output tokens while increasing actual
API spending. The live dollar cap and separate dollar measurements are therefore
essential when assessing the user's cost-efficiency hypothesis.

The payoff schedule is observable: bank and workers see the maximum terminal
reward, normalized-score rule, token price and rent. The current wealth alone
does not tell a decision-maker whether an additional test or implementation is
worth its cost. Private reference answers remain outside these observations.

## Why grants replace bids

A transfer between participants cannot increase total society wealth. Thus a
high bid is no longer a natural proxy for the society's objective. The bank
instead evaluates an explicit plan and requested resource budget. It controls
capital allocation; local agents control method, recruitment and stopping.
The deterministic scheduler checks legality and enforces round boundaries; it
does not secretly choose useful workers, inspect the reference answer, or decide
how a team solves a task.

The bank is intentionally a central bottleneck with constant bandwidth. There
is no globally informed allocator behind it. A queue of every agent's complete
proposal would quietly reintroduce unbounded context and proposal-generation
cost. Here, admission slots are rotated before the bank call. Ineligible
proposals are ignored even if a model generates one. Rotating pairings provide
limited discovery without a model reading an entire directory. Scheduling is
fair among a stable eligible roster; changes to the roster can affect wait time.

## Sponsored entry and recursive company management

Initially no agent has spending rights. Admission is metadata-only: the runtime
chooses a bounded set without asking every waiting model to think. The bank pays
only admitted applications, within an episode-wide application cost ceiling.
Bank reviews are metered separately. This ceiling is not counted as another
asset and does not reserve capital against subsequent grants.

An application keeps one plan, requested amount, latest feedback, and attempt
count. Reviewers approve, request revision, or reject. The initial attempt and at
most `max_revisions` further submissions are allowed per application. A bank
rejection closes that agent's automatic bank route for this episode. Managers can
still sponsor its employment application. Successful grants clear the application
and permit later top-ups, subject to ordinary admission limits.

A funded manager can open a bounded, expiring job offer. Opening it authorizes a
screening call for one matched applicant per round; it must have capital for both
screening/review and the advertised working allowance. Candidates can decline.
A consenting applicant can replace an outstanding bank proposal with a local
employment proposal. No model can solicit arbitrary unfunded conversations.

Every funded agent can be a manager, with the same direct-child cap. Local child
proposals and employment applications share the manager's bounded review batch.
A child funds its own proposal when able; a parent-requested revision can be
sponsored by that parent if the child cannot afford another call. A broke child
with no authorized revision waits for a parent allocation rather than falling
back to the bank. In every case, the generation's payer is explicit in the log.

The bank's brief and each reviewer's feedback give strategic direction. Feedback
is bounded, paid model output, not a hidden task oracle. The scheduler enforces
legality and accounting; it does not choose which proposals are promising.

## What is shared and what is local

The optional bank board is a bounded strategic work registry, not a roster of
roles. Each task contains an ID, scope, acceptance check, status and feedback;
ownership is recorded by the runtime only after a valid grant. The bank can
publish a new proposal's task before approving it in the same response. Describing
an unowned task as claimed merely records intent: it stays open without a grant.
Duplicate task-ID approvals become actionable revision feedback without moving
money. Semantic overlap across IDs remains a model-review limitation.

The bank can retain/update a claim, complete it, block it, or reopen it with
feedback. It cannot silently drop active claims in a replacement board. Completed
and reopened tasks stop the former team's work; its remaining money still belongs
to that team until returned or spent in an authorized formation/application.
Closed teams release claims. Marking a board task done does not pay a reward.

Workers start without specialized roles. Strategies are bounded, editable local
state, and spawning lets the parent propose a child's initial strategy. Task
assignments and specialization can change without introducing new role classes
or changing the scheduler. A recruitment tree alone is not emergence: evidence
would require model-chosen division of labor, adaptation and better outcomes.

Team money has exactly one ledger balance. Each member holds rights to spend part
of it. The sum of rights equals the wallet at every barrier. Hiring and spawning
cannot create money, and a child cannot spend its parent's rights without a
delegation. Each parent can reclaim its children's unused rights and can request
additional rights from its own parent. A manager has the same child cap as any
other member, rather than privileged access to every worker's context.

Leaf-only departure avoids silently moving a recruiter's descendants or
reattaching them in a way that breaks the degree bound. Internal agents can sleep;
the root can dissolve the team. Subtree transfer with descendant consent is a
useful later extension. Current membership changes also clear the departing
agent's bounded task memory/inbox and its old scratchpad slot.

Full-team scratchpads are stored but never pasted into a prompt. Each author
owns one bounded slot. Exact-key reads are capped; local summaries can carry
references to evidence farther down the tree. Public trails are short, expiring
manager publications, visible during formation. They are claims with provenance,
not success signals from the private grader. Quality-based pheromone reinforcement
is deliberately not fabricated in the demo.

## Scalability claims and limits

With constants `m` bank contacts, `q` peer degree, `b` direct children,
`r` local review contacts, `s` work turns, `C` input bytes and `L` output tokens,
all per-agent context and communication surfaces remain bounded. A conservative
per-agent call bound is `s + 3`: formation/application, at most one additional
sponsored revision/application, one local review, and `s` work calls. Waiting
unfunded agents make **zero** calls unless admitted to a sponsored application.
At most `m` bank applications and one bank review call occur per round, plus one
initial planning call when the board is enabled and one terminal evaluator call
for a research task. Each local manager can sponsor at
most one new external applicant per round.

Overall model work is bounded by `O((N_active*(s+3) + m) * (C+L))` per round;
it is not constant total computation. Host pairing/admission include
`O(N log N)` sorting; local child lookup uses bounded adjacency, not all-pairs
agent scans. State is proportional to population and artifacts; archives grow
with calls/time. The offline replay is an inspection tool and loads full recorded
snapshots; models never receive that representation.

A tree can still lose essential details in summaries. Bank approval throughput
is `m` per round, and global integration through a manager can remain a bottleneck.
Grant duration, degree, context size and output size are experimental parameters.
Tasks whose problem/reference/final answer cannot fit configured bounds need a
different environment with paged task access; silently truncating them is invalid.
The Commit0 adapter supplies paged access and container commands for source and
visible tests. Private team branches share code through a public artifact at round
boundaries, admitting at most `m` changed branches by rotating priority. Base hashes
prevent silent overwrites; a conflicting branch is retained. A source syntax
precheck rejects the whole proposed update before mutation, with bounded repair
feedback. It parses source without executing it; runtime compatibility and behavior
remain the isolated grader's responsibility. Synchronization checks
file/byte bounds transactionally before updating an untouched file. New teams inherit
the public code. Test scores never determine merge selection. The latest public
checkpoint is automatically selected at the deadline; an early explicit submission
selects a root branch before final grading. A changed checkpoint receives one full
original-suite check per round, plus agent-requested checks and final grading.

Agents retain two bounded tool receipts and an optional excerpt of a bounded team
notes file. Verified counters, current-test status, and distinct failure summaries
make repeated inspection and stale test claims visible. Bank observations contain
only a bounded batch of progress summaries and one public-test summary, not every
worker's history. With the board enabled, existing owners rotate through those
progress slots; the fixed-size board carries persistent scopes and feedback.
Bad JSON yields feedback on the next ordinary turn; there are no
free repair generations. Integration, grading, and artifact export are deterministic
runtime services, not another globally informed model orchestrator. Coding artifacts
are shared across roots only at round boundaries; work conversations stay local.

This bounds model context, not all host computation. Branch discovery, synchronization,
stored artifacts and logs grow with team count; test costs grow with project complexity.
Even with bounded merge admission, distributing the new checkpoint is population-wide
bookkeeping. Source limits are 128 Python files, 64 KB per file, and 2 MB per branch;
notes are 16 KB. Tool-receipt sizes are configurable constants. The coding preset
uses 64 KB input and 16 KB receipts; the original pilot retains 24 KB/6 KB. This
changes the constant factor, not per-agent asymptotic context growth. These
practical limits must be revisited for larger repositories.
Generation-token wealth excludes Docker CPU/time costs, which are recorded in
check artifacts and should be included in any broader efficiency comparison.

## Expected ant-colony behavior: hypotheses to test

The tree and caps are **designed constraints**. Interesting emergent behavior would
be which agents become recruiters, which strategies become specialists, which
subtasks attract workers, when teams change, and whether useful discoveries
propagate through local messages and shared notes. Evidence should include:

- Division of labor that persists beyond one scripted task.
- Recruitment following useful discoveries rather than indiscriminate growth.
- Local reports of failed approaches that reduce duplicated work elsewhere.
- Dynamic use of small versus strong models, followed by sleep when marginal
  expected reward falls below token cost.
- Improved externally measured quality/cost at larger populations without
  increasing per-agent context or bank bandwidth.

The current replay reports births, hiring, sleep, degree, depth, population and
token costs by phase. These measure mechanics. They do not establish cooperation,
specialization, or improved problem-solving. The scripted policy is explicitly
insufficient evidence for any of those claims.

## Recommended experiments before making performance claims

Use held-out tasks, several seeds, the same grader and reward schedule, and equal
real-dollar and output-token budgets. Compare one strong agent, one inexpensive
agent, fixed teams, original EoM private wealth, and this shared bank. Account for
bank, coordination, rejected actions and judging costs in every method. Report
quality, wall time, real dollars, net virtual wealth and failed/no-submission runs.

Then ablate sponsored versus broadly subsidized formation, proposal feedback,
revision limits, recursive spawning, hiring, public trails, bank model strength,
grants versus a fixed equal allocation, and optional rent. Sweep population while
holding `m`, `q`, `b`, `C`, and `L` fixed. Match total budgets so spawning more
agents is not confused with allocating more inference. Measure duplicated work
and scratchpad reuse using externally auditable evidence, not agents' self-ratings.

Long-term society learning needs a persistent multi-task runner and a principled
mechanism for preserving successful strategies. This prototype establishes the
bounded economic protocol first; it does not implement model weight training,
distributed execution, or a causal estimate of each member's contribution.
