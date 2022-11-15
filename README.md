[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT)
[![Latest Version](https://img.shields.io/github/v/release/AI-SDC/AI-SDC?style=flat)](https://github.com/AI-SDC/AI-SDC/releases)
[![DOI](https://zenodo.org/badge/518801511.svg)](https://zenodo.org/badge/latestdoi/518801511)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/63d31eeb27ec445f9fa9c5866d8bec0e)](https://www.codacy.com/gh/AI-SDC/AI-SDC/dashboard)
[![codecov](https://codecov.io/gh/AI-SDC/AI-SDC/branch/development/graph/badge.svg?token=AXX2XCXUNU)](https://codecov.io/gh/AI-SDC/AI-SDC)

# AI-SDC

A collection of tools and resources for managing the statistical disclosure control of trained machine learning models.

## Content

* `attacks` : Contains a variety of privacy attacks on machine learning models, including membership and attribute inference.
* `docs` : Contains Sphinx documentation files.
* `example_notebooks` : Contains short tutorials on the basic concept of "safe_XX" versions of machine learning algorithms, and examples of some specific algorithms.
* `examples` : Contains examples of how to run the code contained in this repository:
  - How to simulate attribute inference attacks `attribute_inference_example.py`.
  - How to simulate membership inference attacks:
    + Worst case scenario attack `worst_case_attack_example.py`.
    + LIRA scenario attack `lira_attack_example.py`.
  - Integration of attacks into safemodel classes `safemodel_attack_integration_bothcalls.py`.
* `preprocessing` : Contains preprocessing modules for test datasets.
* `risk_examples` : Contains hypothetical examples of data leakage through machine learning models as described in the [Green Paper](https://doi.org/10.5281/zenodo.6896214).
* `safemodel` : The safemodel package is an open source wrapper for common machine learning models. It is designed for use by researchers in Trusted Research Environments (TREs) where disclosure control methods must be implemented. Safemodel aims to give researchers greater confidence that their models are more compliant with disclosure control.
* `tests` : Contains unit tests.

## Documentation

Documentation is hosted here: https://ai-sdc.github.io/AI-SDC/

---

This work was funded by UK Research and Innovation Grant Number MC_PC_21033 as part of Phase 1 of the DARE UK (Data and Analytics Research Environments UK) programme (https://dareuk.org.uk/), delivered in partnership with HDR UK and ADRUK. The specific project was Guidelines and Resources for AI Model Access from TrusTEd Research environments (GRAIMATTER).­ This project has also been supported by MRC and EPSRC [grant number MR/S010351/1]: PICTURES.

<img src="docs/source/images/UK_Research_and_Innovation_logo.svg" width="20%" height="20%" padding=20/> <img src="docs/source/images/health-data-research-uk-hdr-uk-logo-vector.png" width="10%" height="10%" padding=20/> <img src="docs/source/images/logo_print.png" width="15%" height="15%" padding=20/>
