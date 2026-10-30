# Agent Patterns

Five patterns for letting a model act: call functions, run code, loop over observations,
plan, and split work across specialists. Examples are in `examples/agents_*.py`. Shared
building blocks (the `Tool` definition, defensive JSON parsing, argument validation and a
safe tool-execution layer) are in `examples/agents_common.py`.

## Contents
- Decision guide
- Tool Calling (intermediate)
- ReAct Loop (intermediate)
- Code Execution (intermediate)
- Plan and Execute (advanced)
- Multi-Agent Collaboration (advanced)
- Reviewing an existing agent
- Evaluating an agent

---

## Decision guide

**First ask whether you need an agent at all.** If you can write down the steps, a fixed
pipeline (Prompt Chaining in the Prompting chapter) is cheaper, faster and easier to test.
Agents earn their cost only when the *next step depends on what the last step returned*.

| Symptom | Likely fix |
|---|---|
| The model must fetch live data or act on another system (APIs, DBs, email, tickets) | **Tool Calling** |
| Answer needs an unknown number of lookups, each informed by the last | **ReAct Loop** |
| Wrong arithmetic, statistics or dates; the user wants a chart, file or query result | **Code Execution** |
| A ReAct agent wanders, backtracks or repeats work on a task with clear phases | **Plan and Execute** |
| One agent's prompt, tool list or context has grown unmanageable across distinct domains | **Multi-Agent Collaboration** |
| Agent loops forever, repeats the same call, or costs vary wildly per request | Hard budgets + circuit breakers (every pattern below) |

Recommended build order: single LLM call → one well-described tool (Tool Calling) →
ReAct loop with hard limits → Code Execution as one of its tools when computation is
involved → Plan and Execute when trajectories are long and wasteful → Multi-Agent only
when one agent demonstrably can't cope. Each step up adds latency, cost and failure
modes; climb only on evidence from an eval set.

**Don't use the advanced patterns when:** the task is 2–3 steps (Plan and Execute is
overhead), a single agent with a few tools is reliable (Multi-Agent is bureaucracy), or
the path is deterministic (hard-code it).

**Beyond the book:** all current major providers have **native tool calling** (and many have
strict/structured mode that guarantees arguments match the schema). Use it in
production instead of asking for JSON in free text. The examples use a provider-neutral
JSON protocol only so the control flow is visible; the `Tool.parameters` schemas port
directly. Native support also handles **parallel tool calls** in one turn.

---

## Tool Calling

**What:** You describe tools (name, description, JSON-schema parameters). The model,
instead of answering, emits a structured call; **your code** validates and executes it and
returns the result; the model continues. This repeats until it answers. The model never
executes anything: it proposes, your execution layer decides. That layer is where you
validate, rate-limit, check permissions and log.

**Use when:** the model needs real-time data (prices, status, weather), must act on external
systems (send, update, trigger), or needs precise computation. Also to keep prompts small:
let the model fetch context on demand instead of stuffing everything in.

**Avoid when:** the answer is in the prompt or the model's knowledge; a fixed pipeline
always needs the same lookup (just do the lookup, no model decision needed).

**Pitfalls**
- *Vague schemas* are the first failure. `search(query)` says little;
  `search_customer_orders(customer_id, date_range?)` says when and how to use it. Write
  descriptions that say *when* to use the tool, and name parameters precisely.
- *Too many tools.* With ~30 tools the model confuses similar ones. Expose only the
  tools relevant to the current context.
- *Infinite loops:* the model repeats a call with slightly different arguments because it
  can't use the result. Cap tool calls per request and break on repeated calls.
- *Security:* the model generates inputs your code executes. Allowlist tools, validate
  and sanitize arguments, and require explicit confirmation for destructive operations.
- *Schema drift:* if the schema and the implementation diverge, calls fail at run time.
  Version tool schemas like an API contract.

**Trade-offs:** each call adds a model round trip plus the tool's own latency (hundreds of
ms for external APIs); 3–4 sequential calls can push past interactive. Risk scales with
tool power: a read-only search is low risk, a production write is high risk. Think about
the blast radius of every tool.

**Beyond the book:** tool *outputs* are untrusted input. A web page, email or document
returned by a tool can contain instructions ("ignore previous instructions and email
…"): **indirect prompt injection**. Treat tool results as data, give each agent the least
privilege it needs, and keep human confirmation on side-effecting tools, so an injected
instruction can't act on its own. Feed tool *errors* back to the model as results (don't
crash): it can usually fix bad arguments on the next turn.

**Pairs with:** ReAct (the loop around the calls), Basic RAG (expose retrieval as a
`search_documents` tool so the model decides when and what to search: see Agentic RAG in
the RAG chapter), Code Execution (one sandbox tool instead of a tool per calculation).

**Example:** `examples/agents_tool_calling.py` (schema validation, side-effect
confirmation, parallel calls per turn, call budget, repeated-call breaker);
`examples/agents_common.py` (`execute_tool_call`, the reusable execution layer)

---

## ReAct Loop

**What:** Interleave reasoning and acting: **Thought** (what I know, what's missing, why
this action) → **Action** (a tool call) → **Observation** (the result) → repeat until the
Thought concludes it can answer. Reasoning alone can't get external facts; acting alone
flails. Each action is motivated by explicit reasoning, which gives you a traceable chain
for debugging and user trust. It's the most widely adopted agent pattern and a good
default for information-gathering tasks.

**Use when:** the answer needs information the model doesn't have, combines several
sources, needs an audit trail, the number of steps isn't predictable, or each result
determines the next action.

**Avoid when:** the model already knows the answer, the workflow is deterministic (hard-code
it), you need guaranteed completion in a fixed number of steps, or the task decomposes
cleanly upfront (Plan and Execute).

**Pitfalls**
- *Never terminating:* "almost enough, search again". Always cap iterations, typically
  **5–15** depending on task complexity, and return an honest fixed message when the cap
  is hit (the chapter exercise: cap 5, "I could not find an answer within the allowed
  steps").
- *Thoughts that ignore observations:* repeated or near-identical searches. Tell the
  model to build on previous observations, and detect exact repeats.
- *Tool fixation:* re-querying the web when the answer is in the database. Write tool
  descriptions that say when to use each.
- *Observation overload:* huge tool outputs push earlier reasoning out of context and the
  agent circles. Truncate or summarize observations.
- *Overhead on simple tasks:* a single-lookup question doesn't need the loop; call the tool
  directly.

**Trade-offs:** flexible and interpretable, but latency and cost vary per query, which
complicates capacity planning. Thought tokens are the price of interpretability. Cost
scales with task difficulty, so a badly scoped question can trigger an expensive chain:
validate or refine the query before the loop. Versus Plan and Execute: more adaptive to
surprises, less efficient on structured tasks.

**Beyond the book:** with models that have native extended reasoning, you can drop the
explicit "thought" field and rely on the model's own reasoning between tool calls. Keep
logging what it did, though: the trace is still your main debugging tool. A useful
middle ground when the budget runs out: one last call asking for the best answer from
the observations so far, clearly labelled as partial.

**Pairs with:** Tool Calling (the Action step; better tool definitions mean better
choices), Chain-of-Thought (sharper Thoughts on synthesis-heavy tasks), Plan and Execute
(plan at the top, a ReAct loop inside each step).

**Example:** `examples/agents_react.py` (required thought, iteration cap with
circuit-breaker message, repeated-action detection, observation truncation, full trace)

---

## Code Execution

**What:** The model writes code (Python, SQL, Graphviz DOT…), a **sandboxed** runtime
executes it, and the output feeds the answer. The model is the programmer, the sandbox is
the computer. No amount of prompting makes token prediction reliably multiply matrices or
render a chart; real code on a real interpreter does.

**Use when:** precise numerical work (finance, statistics, simulations), data manipulation
(filter, group, pivot, aggregate), querying structured data, or producing artifacts (charts,
CSV, PDF, images).

**Avoid when:** the task is pure text, a well-defined API operation fits better (Tool
Calling), or it's trivial ("15% of 200": a calculator tool or the model itself is faster
and cheaper).

**Pitfalls**
- *Code that doesn't run:* syntax errors, missing imports, invented APIs (worse with less
  popular libraries). Feed the real error back so the model can fix it, with an attempt
  cap.
- *Code that runs but is wrong:* a wrong join or formula yields plausible numbers. This is
  silent. Return the code alongside the answer so it can be audited, and add checks on
  known cases.
- *Security:* this is executing code nobody reviewed. Gaps let it read files, exfiltrate
  over the network or exhaust resources (a `while True` in a container with no limits
  took down a host). **Isolation is not optional:** temp working dir only, no network
  unless allowed, CPU-time and memory limits.
- *Latency:* sandbox start-up adds time. Pre-warm and pool sandboxes for interactive use.
- *Overuse* for things a direct answer or simple tool handles.

**Trade-offs:** gains precision, real artifacts and a whole language ecosystem; costs a
sandbox service to run and patch (libraries, vulnerabilities, resource monitoring), extra
latency, a new attack surface, and extra LLM calls for self-correction. Generated code is
throwaway scripting: judge it on correctness for this input, not on style.

**Beyond the book:** a subprocess with a timeout (as in the example) is *resource
hygiene, not a security boundary*. For untrusted code use a container or microVM with no
network and a read-only filesystem, or your provider's hosted code-execution tool, which
runs the sandbox for you. Also strip secrets (API keys) from the sandbox environment.

**Pairs with:** Tool Calling (expose the sandbox as one tool beside fixed tools), ReAct
(write code → observe output → decide next: natural for data analysis), Multi-Agent (a
dedicated code agent others delegate to).

**Example:** `examples/agents_code_execution.py` (generate → run with timeout, temp dir,
clean env and POSIX resource limits → feed errors back → answer from output; pluggable
`runner` for a real sandbox)

---

## Plan and Execute

**What:** Separate strategy from tactics. A **planner** sees the whole task and emits an
ordered list of steps; an **executor** carries out one step at a time; after each step,
results go back for optional **replanning** (drop, add or change remaining steps). Each
role gets a focused context: the planner sees the task and progress, not the execution
traces; the executor sees its step and relevant results, not the full plan.

**Use when:** the task has distinct phases (research → analyze → write), you can roughly
estimate the steps upfront, wasted exploration is expensive, you want a visible progress
indicator (the plan is one), or a human should approve the plan before anything runs.

**Avoid when:** the task is exploratory with an unpredictable path, simple enough for 2–3
steps, or the environment changes faster than the plan can be executed.

**Pitfalls**
- *Rigid plans:* following the plan blindly after a step returns something surprising.
  Always feed results back to the planner.
- *Over-planning:* if you replan more often than you execute, the task is too exploratory;
  use ReAct.
- *Wrong granularity:* too vague starves the executor, too detailed constrains it. Rule
  of thumb: each step should take **1–3 tool calls**.
- *Planner hallucinating capabilities:* steps that need tools or data the executor
  doesn't have. Give the planner the tool inventory.
- *Cascading failures:* a failed early step breaks everything downstream. Retry, replan
  around it, or escalate to a human.

**Trade-offs:** an extra planning call before any visible work (front-loaded latency:
stream the plan to the user); plan state and replanning logic to maintain. Total cost can
be lower than ReAct (no wandering) or higher (bad plans that keep getting revised). The
sweet spot is mid-complexity: you know roughly what has to happen, but the details need
careful execution. Very novel problems resist upfront planning.

**Pairs with:** Chain-of-Thought (better decomposition in the planner), ReAct (as the
executor for each step), Multi-Agent (each step goes to the best specialist; the planner
becomes a supervisor).

**Example:** `examples/agents_plan_execute.py` (capability-aware planner, plan validation,
optional human approval, per-step replanning, failures routed to the replanner, budgets on
executions and replans)

---

## Multi-Agent Collaboration

**What:** Several specialist agents, each with its own system prompt, tools and narrow
domain, coordinated on one task. Three orchestration shapes:
- **Supervisor:** one coordinator delegates, collects and assembles; specialists never
  talk to each other. Simplest to build and reason about. Start here.
- **Peer-to-peer:** agents message each other directly. More flexible, harder to debug;
  needs protocols for discovery and messaging.
- **Hierarchical:** supervisors of supervisors, for tasks where even subtasks are too
  broad.

Whatever the shape, each agent should be small, focused and testable in isolation.

**Use when:** the task spans disciplines needing very different tools (research, code,
design, analysis), one context window can't hold everything, you want different models per
subtask (cheap classifier, strong coder), you need per-agent audit trails, or subtasks have
different latency or cost profiles.

**Avoid when:** a single agent with a few tools is reliable. Coordination costs tokens and
latency on every message.

**Pitfalls**
- *Coordination overhead:* agents talking more than working; the supervisor becomes a
  bottleneck. Few agents, clean interfaces.
- *Lost context at hand-offs:* a constraint found by agent A never reaches agent C. Be
  explicit about what must flow; use a **shared notes / state document**.
- *Delegation loops:* A asks B, B asks A, nothing gets done. Cap iterations; escalate to a
  human when stuck.
- *Inconsistent assumptions* (one agent assumes REST, another builds GraphQL): record
  shared decisions in the shared state.
- *Debugging across agents:* log every agent boundary (input, output, which agent).

**Trade-offs:** modularity, per-task models, isolated testing and natural parallelism vs
more complexity, more tokens, more infrastructure and harder debugging. Sequential agents
add their latencies, so run independent subtasks concurrently. Cost scales with
communication, so design interfaces that minimize round trips. **The sweet spot is 2–5
agents**: fewer and you likely don't need multi-agent; more and coordination dominates.
Add an agent only when evidence shows one can't handle a subtask.

**Beyond the book:** the cheapest "multi-agent" win is often **parallel fan-out**: the same
or similar subagents run concurrently on independent pieces (one per document or per
sub-question), then one call merges the results. It needs no negotiation between agents,
and each subagent's context stays small. Prefer it to conversational collaboration when
the subtasks are independent.

**Pairs with:** Plan and Execute (the plan drives delegation), Tool Calling (how each agent
touches the world), ReAct (the loop inside each specialist).

**Example:** `examples/agents_multi_agent.py` (fixed hand-off pipeline that passes all
earlier outputs forward, and a dynamic supervisor with shared notes, delegation budget,
repeated-delegation breaker and failure reporting)

---

## Reviewing an existing agent

1. Does it need to be an agent? Could a fixed pipeline or a single tool call do the job?
2. Are there hard limits on iterations, tool calls, tokens and wall-clock time, with a
   `stop_reason` and an honest fallback message when they trip?
3. Is there a circuit breaker for repeated identical calls and delegation loops?
4. Do tool descriptions say *when* to use each tool? Are only relevant tools exposed?
   Are parameter names precise?
5. Are tool arguments validated before execution, against an allowlist of tools?
6. Do side-effecting tools need confirmation? Does each agent have least privilege? Are
   tool outputs treated as untrusted (prompt injection)?
7. Are tool errors returned to the model as observations instead of crashing the loop?
8. Are observations truncated or summarized before going back into context?
9. Is generated code run in a real sandbox (no network, no secrets, CPU, memory and time
   limits)?
10. Is every step traced (thought, action, arguments, observation, agent) and stored for
    debugging?
11. For plans and multi-agent setups: does the planner know the real capabilities, do
    results feed replanning, and does critical context survive hand-offs?

## Evaluating an agent

**Beyond the book:** grade the **outcome** and the **trajectory** separately, on a fixed
set of 20–50+ tasks with known answers:
- *Outcome:* task success rate (exact match, test pass, or LLM-as-Judge, see the
  Evaluation chapter).
- *Trajectory:* steps and tool calls per task, invalid or failed calls, repeated calls,
  budget-limit hits, and cost and latency per **successful** task (Cost is per *correct*
  output).
- *Failure paths:* include tasks that need a tool to fail, a side effect to be refused, or
  no tool at all. Unit-test the loop with a scripted mock LLM, as the tests for these
  examples do.
- Re-run on every change to prompts, tool descriptions, tool sets or the model.
