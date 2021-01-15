# Reliability and Resilience Assessment Module

Reliability and resilience assessment for more power electronic power systems.

This package is developed and maintained by the energy and electrical research center @ Jinan University.

If you are interested in this work, please feel free to contact us via the following e-mails:

jagerzeng@gmail.com

Date: 15 Jan 2021 

## 0. Introduction
This module is developed to model and assess the reliability and resilience of more power electronic power systems, especially in the face of the emerging power electronic devices and systems.

This package follows classical bottom-up reliability assessment procedures, i.e., equipment modelling, system modelling and analysis. 


## 1. Reliability Modelling
### 1.1 Generators 
### 1.2 Loads
### 1.3 Transmission Lines
#### 1.3.1 AC lines
#### 1.3.2 DC lines
### 1.4 Convertors
### 1.5 Energy Storage Systems

## 2. Resilience Modelling
This part will be updated later, responding to the unexpected failure of equipment, cascading failures, etc.

## 3. Power Flow Analysis
Based on KCL and KVl, using the current injection power flow and branch power flow methods.
### 3.1 AC power flow
using Pypower, i.e., the python version of Matpower. 
### 3.2 DC power flow
Using Pypower 
### 3.3 Hybrid AC/DC power flow
using the method in Hynet

## 4. Sampling Methods
using random sampling methods in numpy

## 5. Reliability Indexes
Extending classical reliability indexes

## 6. Non-sequential Analysis
