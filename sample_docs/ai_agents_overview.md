# AI Agents: An Overview

An AI agent is a software system that perceives its environment, decides what to do, and acts to reach a goal. Modern **autonomous agents** are usually built around a large language model that can plan, call tools, and read the results of those tools before deciding on the next step.

## How an agent works

An agent repeats a simple loop. It observes the current state, reasons about the goal, picks an action such as a web search or a database query, and then observes the outcome. This loop continues until the task is finished or the agent decides it is stuck. Because the agent chooses its own actions, it can handle tasks that were never written down as a fixed script.

## Tools and memory

Tool use is what separates an agent from a plain chatbot. A tool can be a calculator, a code interpreter, a search engine, or an internal API. Agents also keep memory. Short-term memory holds the current conversation, while long-term memory is often a vector database that stores earlier results so the agent can retrieve them later.

## Planning

Planning means breaking a large goal into smaller steps. Some agents plan everything first, and others plan one step at a time and adjust after every observation. Good planning reduces wasted tool calls and helps the agent recover when a step fails.

## Risks

Autonomous agents can make mistakes with real consequences, for example by deleting files or sending the wrong email. Teams therefore limit what tools an agent may use, log every action, and ask a human to approve risky steps.
