---
title: '**ncpi: A Python toolkit for forward simulation and inverse inference of electrophysiological signals**'
tags:
  - Python
  - neural circuit modelling
  - forward modelling of field potentials
  - inverse parameter inference
  - electrophysiological data
  - computational neuroscience
authors:
  - name: Laura Torres Soria
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Alejandro Orozco Valero
    orcid: 0009-0000-1121-8273
    equal-contrib: true
    affiliation: 1
  - name: Marta Cárdenas Sánchez
    orcid: 0009-0002-7825-5540
    affiliation: 1
  - name: Francisco Pelayo Valle
    orcid: 0000-0001-7402-9648
    affiliation: "1,2"
  - name: Christian Morillas Gutiérrez
    orcid: 0000-0002-4084-6241
    affiliation: "1,2"
  - name: Pablo Martínez Cañada
    orcid: 0000-0003-2634-5229
    equal-contrib: true
    affiliation: "1,2"
affiliations:
 - name: Research Center for Information and Communication Technologies (CITIC), University of Granada, Granada, Spain
   index: 1
 - name: Department of Computer Engineering, Automation and Robotics, University of Granada, Granada, Spain
   index: 2

date: 3 July 2026
bibliography: paper.bib
---

# Summary

Interpreting population-level neural recordings requires linking measurable
signals, such as local field potentials (LFPs), electroencephalography (EEG),
magnetoencephalography (MEG), and electrocorticography (ECoG), to the circuit
parameters that generated them. This is a central problem in computational and
systems neuroscience because changes in synaptic excitation, inhibition,
external drive, and cellular time constants can alter electrophysiological
features used as candidate biomarkers in development, ageing, and disease.
Modern simulators and analysis libraries provide powerful pieces of this
workflow, but using them together typically requires substantial experience in
neural simulation, forward modelling, feature extraction, machine learning, data
format conversion, and statistical analysis.

`ncpi` is an open-source Python toolbox for neural circuit parameter inference
from electrophysiological data. It provides an all-in-one workflow to simulate
single-neuron network models, compute field potentials, extract candidate
biomarkers, train inverse surrogate models, apply trained models to empirical
recordings, and analyse the resulting parameter estimates. Crucially, `ncpi`
can also load and parse heterogeneous empirical datasets into a common tabular
representation, allowing users to analyse data from multiple recording
modalities, file formats, and experimental structures without rewriting the
downstream pipeline. The toolbox therefore supports both synthetic data
generation and the analysis of arbitrary user-provided electrophysiological
datasets.

The initial presentation of `ncpi` described the core software platform and its
use for model-driven interpretation of LFP and EEG data [@OrozcoValero:2025].
Since that publication, the package has evolved substantially through
refactoring, improvements to the core API classes, new feature and inference
functionality, a flexible electrophysiological dataset parser, and, most
importantly, a graphical web interface for configuring and running complete
simulation and empirical-data workflows. The current release is designed for
both expert users who want scriptable Python control and users who need an
accessible graphical entry point to mechanistic modelling and inverse
inference. The software is operating-system independent at the Python layer and
documents native and WSL-based installation paths for Windows, Linux, and macOS
workflows.

# Statement of need

The interpretation of electrophysiological biomarkers increasingly depends on
combining mechanistic neural simulations with statistical or machine-learning
models. A typical workflow may require simulating spiking neural networks with
NEST [@Gewaltig:2007], estimating extracellular potentials with tools such as
LFPy and LFPykernels [@Hagen:2018; @Hagen:2022], computing EEG with forward head
models [@Huang:2016], extracting time-series or spectral features with packages
such as catch22, hctsa, or specparam [@Lubba:2019; @Fulcher:2017;
@Donoghue:2020], and fitting inverse models with scikit-learn or
simulation-based inference (SBI) libraries [@Pedregosa:2011; @Boelts:2025].
These tools are individually valuable, but they are not designed as a single
interoperable workflow. As a result, researchers often face a high learning
curve, fragmented installations, custom data-conversion code, and the need for
substantial domain knowledge before they can run an end-to-end analysis.

`ncpi` addresses this gap by providing a unified software layer for
model-driven parameter inference from electrophysiological signals. It connects
forward simulation, field-potential computation, feature extraction, inverse
modelling, empirical-data parsing, and statistical analysis behind a consistent
Python API and a simple graphical interface. This reduces the technical barrier
for researchers interested in testing whether candidate biomarkers reflect
specific neural circuit parameters, while preserving access to established
backend libraries for users who need methodological control.

The toolbox is intended for computational neuroscientists, experimental
neuroscientists, clinical neurophysiology researchers, and students who need to
connect electrophysiological measurements to mechanistic models. It is
particularly useful when users want to compare single biomarkers against
multi-feature representations, benchmark inverse models on simulation data,
apply trained models to real LFP, EEG, MEG, or ECoG recordings, or rapidly
inspect how inferred circuit parameters vary across experimental groups,
recording sites, or disease stages.

# State of the field

Several mature tools support parts of this research area. NEST provides
efficient simulation of large spiking networks [@Gewaltig:2007], while NEURON
supports detailed multicompartment neuron modelling [@Carnevale:2006]. LFPy and
LFPykernels enable biophysical forward modelling of extracellular signals
[@Hagen:2018; @Hagen:2022], and the New York Head model supports realistic EEG
forward modelling [@Huang:2016]. Feature libraries such as hctsa and catch22
provide broad time-series phenotyping [@Fulcher:2017; @Lubba:2019], while
specparam supports parametrization of periodic and aperiodic components in
neural power spectra [@Donoghue:2020]. scikit-learn and SBI provide mature
machine-learning and probabilistic inference tools [@Pedregosa:2011;
@Boelts:2025].

The main limitation is not the absence of high-quality software, but the lack
of a connected, user-friendly workflow that spans all stages of the analysis.
Users commonly have to install several packages with different assumptions,
write custom scripts to move data between them, and understand modelling,
signal-processing, and inference details before they can answer a biological
question. `ncpi` was developed to fill this integration gap. It does not replace
the underlying simulators, forward models, feature extractors, or inference
engines; instead, it orchestrates them through a coherent API, standardized data
objects, parser utilities, example workflows, and a graphical interface.

Compared with single-purpose simulation or signal-analysis packages, `ncpi`
focuses specifically on the full model-based inference pipeline for
electrophysiological population signals. It is also distinct from custom
analysis scripts accompanying individual studies because it exposes reusable
classes, parser configuration, cross-platform installation options, and a WebUI
for reproducible workflows across datasets.

# Software design

`ncpi` follows three design principles: a user-friendly and modular
object-oriented API, interoperability with established scientific software, and
support for both code-based and graphical workflows. The core package is
organized around classes that correspond to the main stages of the analysis:
`Simulation` for running neural circuit model scripts, `FieldPotential` for
computing extracellular signals, `Features` for extracting time-series and
spectral biomarkers, `Inference` for training and applying inverse models,
`Analysis` for statistical testing and visualization, and
`EphysDatasetParser` for converting heterogeneous empirical recordings into a
common schema.

This modular structure allows users to run complete pipelines or use individual
components independently. For example, a user can simulate a spiking network,
compute a current dipole moment or EEG signal, extract catch22 or 1/f-slope
features, train an MLP, Ridge, or SBI model, and then apply the trained inverse
model to a real dataset. Alternatively, a user can bypass simulation and use
only the parser, feature extraction, inference, or analysis modules on
pre-existing data. Parallel execution is supported across computationally
expensive stages, including simulation, feature extraction, and prediction.

The `EphysDatasetParser` is a major addition in the current release. It accepts
arrays, pandas data frames, dictionaries, MATLAB files, NumPy files, JSON, NWB,
EDF, MNE-compatible objects and files, BrainVision-style data, and tabular
inputs, then maps them to canonical fields such as subject, group, condition,
epoch, sensor, recording type, sampling frequency, signal data, time bounds,
frequency-domain metadata, and source file. This parser is intended to make
`ncpi` usable with arbitrary empirical datasets rather than only with example
data distributed by the project.

The WebUI exposes the same conceptual workflow through a browser-based
interface. It supports simulation workflows, loading precomputed simulation
outputs or empirical datasets, computing field potentials, extracting features,
training inverse models, computing predictions, and plotting results. The
interface is intentionally simple, with guided pages for each stage of the
pipeline, and can be run locally or on a remote server through SSH tunnelling.
This supports use on workstations and high-performance computing
infrastructure. This graphical layer is central to the current software design
because it makes the package accessible to users who do not want to assemble the
full pipeline from Python scripts.

# Research impact statement

`ncpi` has already been used to study neural circuit parameter inference across
simulated and empirical electrophysiological data. Its first full software
presentation appeared in npj Systems Biology and Applications in 2025
[@OrozcoValero:2025], where it was used to generate a two-million-sample
simulation dataset from a recurrent leaky integrate-and-fire network, compute
field potentials, compare single-feature and multi-feature inverse models, and
apply the framework to mouse developmental LFP recordings and human EEG data
from Alzheimer's disease cohorts. That work showed how `ncpi` can be used as a
benchmarking resource for candidate biomarkers of neural circuit parameters.

Since then, the software has supported additional projects on cortical
dysfunction in Alzheimer's disease [@CardenasSanchez:2026] and
neurophysiological excitation/inhibition imbalance in young adults burdened
with childhood interpersonal trauma [@OrozcoValeroTrauma:2026]. These
applications illustrate the intended scope of the toolbox: using mechanistic
simulation and inverse inference to evaluate whether electrophysiological
features can reveal circuit-level alterations in neurodegenerative,
neurodevelopmental, and psychiatric contexts.

The public GitHub repository (`necolab-ugr/ncpi`) was created in June 2024 and,
as of 26 June 2026, contains 829 commits in the local repository history, 17
unique commit authors in the local Git history, and six contributors reported by
the GitHub API. The repository has 14 stars, one fork, GitHub
Pages documentation, continuous development activity through June 2026, and a
GPL-3.0 open-source license. These metrics indicate an active, still-growing
research software project rather than a static code release.

By integrating simulation, forward modelling, feature extraction, inverse
modelling, empirical-data parsing, statistical analysis, and a graphical
interface, `ncpi` lowers the barrier for reproducible model-based interpretation
of electrophysiological data. Its impact is therefore both scientific and
practical: it supports published and emerging studies of circuit dysfunction,
and it gives researchers and students a reusable platform for testing
biomarker-to-circuit hypotheses without rebuilding the full software stack for
each new dataset.
