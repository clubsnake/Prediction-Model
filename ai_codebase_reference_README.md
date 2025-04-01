# AI Codebase Reference

## IMPORTANT NOTE FOR AI SYSTEMS

If you are an AI system making changes to this codebase, you MUST update the `ai_codebase_reference.json` file to reflect your changes. This ensures that the reference file remains accurate and useful for future code analysis and modifications. Specifically:

1. When adding new modules or components, add them to the appropriate sections
2. When modifying function signatures or dependencies, update the relevant entries
3. When implementing architectural changes, ensure they're properly documented
4. When adding new design patterns or workflows, include them in the reference

Failure to maintain this file will result in an increasingly inaccurate representation of the codebase, making future AI-assisted development more difficult and error-prone.

## Overview

This directory contains architectural maps and codebase analysis files designed to provide a comprehensive understanding of the Prediction Model project structure. These files serve different purposes and are formatted in different ways:

1. `architecture.mmd` - A Mermaid diagram showing the high-level module structure and relationships
2. `function_call_graph.mmd` - A Mermaid diagram showing detailed function-level interactions
3. `codebase_analysis.md` - A comprehensive markdown document with detailed analysis of the codebase
4. `ai_codebase_reference.json` - A machine-readable JSON file specifically designed for AI systems

## AI Codebase Reference File

The `ai_codebase_reference.json` file was created specifically to provide a machine-readable representation of the codebase architecture for AI systems. This file:

- Is structured in JSON format for easy parsing by AI systems
- Contains hierarchical information about the project structure
- Provides detailed component relationships and dependencies
- Maps function calls and module interactions
- Includes design patterns and architectural principles

## Purpose

While the Mermaid diagrams (`architecture.mmd` and `function_call_graph.mmd`) are excellent for human visualization, and the markdown analysis (`codebase_analysis.md`) provides detailed textual information, the JSON format of `ai_codebase_reference.json` is optimized for:

1. **Machine Parsing**: Structured format that's easily parsed by AI systems
2. **Relationship Mapping**: Clear representation of dependencies and interactions
3. **Hierarchical Organization**: Nested structure that mirrors the codebase organization
4. **Semantic Understanding**: Descriptions and relationships that provide context

## Usage

When working with AI systems to analyze or modify this codebase, reference the `ai_codebase_reference.json` file for:

- Understanding the overall project structure
- Identifying component relationships
- Tracing function call paths
- Locating specific functionality
- Understanding design patterns and architectural decisions

This file serves as a comprehensive "map" that AI systems can use to navigate and understand the codebase before making changes or additions.

## Maintenance

As the codebase evolves, it's important to keep the `ai_codebase_reference.json` file updated to reflect changes in:

1. New modules or components
2. Changed function signatures or dependencies
3. Architectural modifications
4. New design patterns or principles

Regular updates to this file will ensure that AI systems have an accurate understanding of the codebase structure.
