# theme-explorer

Exploratory tool for user feedback

## Project Organisation

```text
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for this project.
├── data
│   ├── intermediate   <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
││
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         theme-explorer and configuration for tools like ruff
│
├── output             <- Generated outputs like figures and tables as HTML, PDF, LaTeX, etc.
│
├── project_config     <- Global config settings for the project, eg visualisation settings
│
└── src
    └── theme_explorer        <- Source code for use in this project.
        │
        ├── __init__.py                     <- Makes theme-explorer a Python module
        ├── llm <- module for LLM interactions
        └── utils <- module for utilities
        
```

--------
