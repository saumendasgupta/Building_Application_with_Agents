# Building Applications with AI Agents


---

## Overview

This repository provides a **unified, multi-framework platform** for designing, implementing, and evaluating AI-powered agents. By separating **scenario definitions** from **framework-specific code**, we enable:

* A **single spec** for each scenario (under `src/common/scenarios/`).
* **Parallel implementations** in LangGraph, LangChain, Autogen (and more).
* A **shared evaluation harness** to compare outputs across frameworks.
* **Built-in observability** (Loki logging & OpenTelemetry/Tempo).
* **Unit tests** for core utilities and telemetry setup.

Whether you’re building an e-commerce support bot, an IT support desk assistant, a voice agent, or anything in between, this codebase helps you scale from prototype to production coverage—while maintaining consistency and reusability.

---

## Book

This repository accompanies the O’Reilly Media book
[**Building Applications with AI Agents: Designing and Implementing Multi-Agent Systems**](https://www.oreilly.com/library/view/building-applications-with/9781098176495/).
