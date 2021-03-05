# Evolving Systems Library

[![DOI](https://zenodo.org/badge/276922055.svg)](https://zenodo.org/badge/latestdoi/276922055)

Python implementation of evolving algorithms.

## About

This is a library project to facilitate the use of evolutionary algorithms.
Initially, the algorithms added to the library perform data clustering.

## Project Organization

    ├── LICENSE
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── algoritms          <- Trained and serialized models, model predictions, or model summaries.
    │   ├── __init__.py    <- Makes algoritms a Python module.
    │   └── denstream      <- DenStream algorithm folder.
    │   └── sostream       <- SOStream algorithm folder.
    │   └── macro_sostream <- Macro SOStream algorithm folder, modification of SOStream.
    │   └── autocloud      <- Autocloud algorithm folder.
    │
    ├── metrics            <- Scikit-learn purity, accuracy, recall and f1-score metrics.
    │   ├── __init__.py    <- Makes metrics a Python module.
    │   └── Metrics.py
    │
    ├── notebooks examples <- Jupyter notebooks with usage examples.
    │
    └── setup.py           <- Makes project pip installable.


## Would you like to contribute?
<p><small>The project is still under construction, so if you like it enough to collaborate, just let us know or simply create a Pull Request.</small></p>
