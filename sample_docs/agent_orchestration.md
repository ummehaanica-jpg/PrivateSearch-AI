# Agent Orchestration Patterns

Orchestration is the layer that decides which agent or tool runs next, what it receives as input, and how its output is combined with the work of others. Without orchestration, a group of agents is just a group of programs that happen to run at the same time.

## Supervisor pattern

A supervisor agent receives the user request, splits it into subtasks, and routes each subtask to a specialist worker. The supervisor then merges the results into one answer. This pattern is easy to debug because every decision passes through one place.

## Pipeline pattern

In a pipeline, the output of one agent becomes the input of the next. A research agent might gather sources, a writing agent drafts a summary, and a review agent checks the draft. Pipelines are predictable but they cannot easily adapt when an early step goes wrong.

## Handoffs and routing

A handoff transfers control from one agent to another together with the conversation state. A router inspects the request and chooses the best agent, which keeps each agent focused on a narrow job and keeps prompts short.

## State, retries and observability

Orchestrators store shared state, retry failed steps, and enforce time and cost limits. Tracing every call is essential, since a failure in a multi-step workflow is hard to diagnose without a complete record of what each agent saw and did.
