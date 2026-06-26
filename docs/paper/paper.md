---
title: '**ncpi: A Python toolkit for forward simulation and inverse inference of electrophysiological signals**
'
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

## Summary

`ncpi` is an open-source Python toolbox that unifies forward modelling of field potentials and inverse parameter 
inference into a single workflow for the neuroscience community. The package provides a unified workflow to simulate, 
analyse, and infer parameters of neural circuit models from electrophysiological population signals. By integrating 
state-of-the-art methods for forward modelling of extracellular field potentials with machine-learning-based inverse 
modelling approaches, `ncpi` enables users to move from mechanistic neural circuit simulations to data-driven parameter 
inference within a single software environment. A central goal of `ncpi` is to make complex model-based analysis 
pipelines accessible to a broad neuroscience community. The toolbox supports multiscale electrophysiological signals, 
ranging from local field potentials (LFPs) to non-invasive electroencephalography (EEG) and magnetoencephalography (MEG), 
allowing users to investigate how neural circuit parameters shape measurable population-level activity across spatial 
scales. The software can be used to generate synthetic datasets, extract candidate electrophysiological biomarkers, 
train inverse models, and estimate circuit parameters from empirical recordings.

This release introduces a graphical web interface that allows users to configure and run complex forward and 
inverse modelling workflows with minimal programming experience. Through an interactive browser-based interface, 
users can launch simulations, define parameter ranges, compute electrophysiological features, train inference models, 
and apply them to experimental data with only a few clicks. The graphical interface is designed to complement the 
Python API, enabling both expert users who require scriptable control and non-specialist users who need an accessible 
entry point to mechanistic modelling and parameter inference. `ncpi` is operating-system independent and can be run 
locally, on servers, or on high-performance computing facilities. The toolbox is designed to exploit the available 
multiprocessing capabilities of the host system, making it suitable for computationally demanding simulation and 
inference pipelines. To support usability, reproducibility, and teaching, `ncpi` includes extensive documentation, 
video tutorials, and Jupyter notebooks that guide users through the main steps of the workflow, from forward 
simulations to inverse inference on electrophysiological data.

