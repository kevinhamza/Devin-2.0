<p align="center">
  <pre>
                                       ██████╗ ███████╗██╗   ██╗██╗███╗   ██╗
                                       ██╔══██╗██╔════╝██║   ██║██║████╗  ██║
                                       ██║  ██║█████╗  ██║   ██║██║██╔██╗ ██║
                                       ██║  ██║██╔══╝  ╚██╗ ██╔╝██║██║╚██╗██║
                                       ██████╔╝███████╗ ╚████╔╝ ██║██║ ╚████║
                                       ╚═════╝ ╚══════╝  ╚═══╝  ╚═╝╚═╝  ╚═══╝
  </pre>
</p>
<h1 align="center">Devin AGI Project</h1>
<p align="center">
  <em>An Autonomous General Intelligence for Complex Software, Cybersecurity, and Robotics Tasks</em>
</p>

<p align="center">
    <a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.9+-blue" alt="Python Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/tests-100%25-brightgreen" alt="Tests"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

**Devin** is a proof-of-concept Autonomous General Intelligence (AGI) designed to operate as a sophisticated software engineer, cybersecurity analyst, and automation specialist. This project integrates a powerful cognitive architecture with a vast suite of tools, enabling the agent to understand complex goals, formulate multi-step plans, and execute them across a wide range of digital and physical environments.

## ✨ Core Features

This project is a comprehensive ecosystem of integrated capabilities:

-   **🧠 AI & Cognitive Core**
    -   **Advanced AI Agent:** Central orchestrator (`AIAgent`) capable of using multiple LLM providers (OpenAI, Gemini, Perplexity).
    -   **Value-Aligned Decision Making:** A sophisticated `UtilityFunction` ensures every action is measured against core principles like safety, efficiency, and task completion.
    -   **Ethical Guardrails:** A programmable `EthicalConstraint` system provides a non-bypassable moral compass to prevent harmful actions.
    -   **Recursive Self-Improvement:** Includes modules for `SelfModifyingCodeGenerator` to improve its own code and `KnowledgeDistillation` to learn from more powerful AI models.

-   **🛠️ Automation & OS Control**
    -   **Desktop Automation:** Controls the keyboard and mouse (`DesktopAutomator`) to operate native GUI applications.
    -   **Web Automation:** Manages a web browser (`WebAutomator`) to perform complex tasks like data scraping, form filling, and site navigation.
    -   **Universal OS Operator:** A cross-platform abstraction layer (`UniversalOSOperator`) to interact with Windows, macOS, and Linux filesystems and system calls.

-   **🛡️ Cybersecurity & Threat Intelligence Suite**
    -   **Pentesting Toolchain:** A full suite of offensive security tools (`PentestingFacade`) for network scanning, web application analysis, and vulnerability exploitation.
    -   **Threat Intelligence Feeds:** Automatically downloads and queries data from `MITRE ATT&CK`, `VirusTotal`, and other IOC feeds for real-time threat context.
    -   **Defensive Capabilities:** Includes an AI-powered `ThreatAnalyzer` to detect sophisticated phishing and a `MalwareSandbox` for dynamic analysis of suspicious files.

-   **☁️ Multi-Cloud Management**
    -   **Unified Cloud Facade:** A single interface (`CloudFacade`) to manage resources (VMs, storage, databases) across AWS, GCP, and Azure.
    -   **Normalized Data:** Automatically translates provider-specific data into a clean, standardized format for easy analysis.
    -   **Live Server Backend:** A dedicated `CloudIntegrationServer` provides a robust API for all cloud operations.

-   **🤖 Robotics & Extended Reality (XR)**
    -   **ROS 2 Integration:** A professional-grade `ROS2Interface` allows for communication and control of modern robots.
    -   **Autonomous Navigation:** A complete robotics stack for perception (`ObjectDetector`), mapping (`SLAMSystem`), and navigation (`AINavigationSystem`).
    -   **3D World Building:** A real-time bridge to the **Unity 3D engine** (`UnityIntegration` & `SpatialComputer`) for programmatic creation and control of metaverse environments.
    -   **Blockchain Identity:** A self-contained `NFTGenerator` for creating and managing digital assets and identities.

## 🏗️ Architecture Overview

Devin operates on a modular, agentic architecture. At its heart is a "Think-Act" loop where the `AIAgent` formulates a plan, which is then vetted by the `UtilityFunction` and `EthicalConstraint` modules before being dispatched by the `ToolExecutor` to the appropriate specialized tool.

For a more detailed breakdown, see the full Architecture Documents.

```mermaid
sequenceDiagram
    participant User
    participant MainLoop (main.py)
    participant AIAgent
    participant Guardian (Utility/Ethics)
    participant ToolExecutor
    participant Tool # (e.g., CloudFacade)

    User->>MainLoop: "Stop all non-prod AWS VMs"
    MainLoop->>AIAgent: Formulate a plan for the goal.
    AIAgent-->>MainLoop: Plan: {"tool": "list_vms", ...}
    MainLoop->>Guardian: Evaluate Plan
    Guardian-->>MainLoop: Plan is Safe & Useful
    MainLoop->>ToolExecutor: Execute this tool call.
    ToolExecutor->>Tool: call list_vms("AWS")
    Tool-->>ToolExecutor: Returns list of VM data.
    ToolExecutor-->>MainLoop: Result: [VM1, VM2, ...]
    MainLoop->>AIAgent: Continue plan with new context (VM list).
    AIAgent-->>MainLoop: Plan: {"tool": "stop_vm", ...}
    MainLoop->>...: (Loop continues)
