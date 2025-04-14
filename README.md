*# TMA4180 Optimization Notes

LaTeX notes and materials for the TMA4180 Optimization course. This repository contains comprehensive course notes with custom styling, mathematical definitions, and algorithm visualizations in both Norwegian and English.

## Prerequisites

- A LaTeX distribution (e.g., TeX Live, MiKTeX)
- The `glossaries` package
- The custom `trymtex` package

## Directory Structure

```
.
├── preamble/
│   ├── bib/
│   │   └── glossary.tex      # Mathematical terminology definitions
│   └── appendix/
│       └── AlgorithmMap.tex  # Visualization of optimization algorithms
```

## Features

- **Custom Environments**
  - Theorem and proof boxes with color-coding
  - Filing and railing boxes for important content
  - Custom coloring and styling definitions

- **Mathematical Content**
  - Comprehensive glossary of optimization terms
  - Algorithm visualization maps
  - Support for both Norwegian and English terminology

- **Topics Covered**
  - Unconstrained Optimization
  - Constrained Optimization (Equality and Inequality)
  - KKT Conditions
  - Linear and Quadratic Programming
  - Optimization Algorithms (Steepest Descent, Interior Point Methods)

## Compilation

To compile the documents:

1. Ensure all prerequisites are installed
2. Run ´´´luatex´´´ with ´´´biber´´´ and ´´´makeglossaries´´´ commands as needed.

## Style Customization

The `trymtex.sty` package provides custom styling including:
- Color-coded theorem environments
- Custom box environments for examples and definitions
- Specialized bibliography and glossary formatting

## Contributing

Feel free to submit improvements or corrections through pull requests.
*