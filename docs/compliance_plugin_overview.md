# Compliance Plugin Overview

This extension adds a modular *Privileged Contrastive MoE Compliance with Pareto Distillation* plugin to whole-body tracking.

- Base task structure and action semantics remain intact.
- Plugin is opt-in via compliance config/flags.
- Teacher uses privileged signals during training only.
- Student uses observable history for deployment.
