# Align-CodeGemma: JAX-Aware Code Generation & Alignment

A comprehensive framework for fine-tuning Google's **Gemma-3** to generate high-quality **JAX-based code** for computational tasks. This project implements advanced alignment techniques, execution-time validation, and reward-driven optimization using **GRPO** (Guided Reinforcement with Prompt Objectives).

## 🎯 Project Overview

This project addresses the challenge of generating domain-specific JAX code by:
- Fine-tuning Gemma-3 4B with specialized alignment techniques
- Implementing real-time code execution validation
- Using reinforcement learning to optimize code quality
- Developing custom metrics for JAX primitive evaluation

## 🚀 Key Features

- **Fill-in-the-Middle (FIM) Training**: Advanced data preprocessing for better code completion
- **GRPO Optimization**: Custom reinforcement learning approach for code generation improvement
- **Execution Validation**: Real-time code testing and validation pipeline
- **JAX-Specific Metrics**: Custom evaluation framework for JAX primitive usage
- **Scalable Architecture**: Docker-based deployment with FastAPI execution server

## 📁 Project Structure

```
Align-CodeGemma/
├── datas/                          # Data preprocessing & tokenization
│   ├── prepare_data.py            # Dataset preparation and format conversion
│   └── tokenize_dataset.ipynb     # Tokenization analysis and optimization
│
├── execserver/                     # Code execution and validation system
│   ├── server.py                  # FastAPI server for code execution
│   ├── routes.py                  # API endpoints for validation pipeline
│   └── utils.py                   # JAX primitive validation utilities
│
├── src/                           # Core alignment and inference logic
│   ├── alignment.py               # FIM-style training data generation
│   ├── inference.py               # CodeGemma model interaction and decoding
│   ├── evaluate.py                # Functional correctness evaluation
│   └── rewards.py                 # Multi-objective reward computation
│
├── Dockerfile                     # Container deployment configuration
├── .gitattributes
├── .gitignore
└── README.md
```

## 🔧 Technical Implementation

### Data Processing (`datas/`)
- **Data Preparation**: Converts raw code datasets into FIM-compatible training formats optimized for JAX syntax
- **Tokenization Analysis**: Implements specialized tokenization strategies for JAX computational graphs

### Execution Server (`execserver/`)
- **Validation Pipeline**: Real-time code execution with security sandboxing
- **API Framework**: RESTful endpoints for code testing and result aggregation
- **JAX Analysis**: Automated detection and validation of JAX primitives

### Core Engine (`src/`)
- **Model Alignment**: FIM-based training data generation with JAX-specific optimizations
- **Inference System**: Advanced decoding strategies with constraint-guided generation
- **Evaluation Framework**: Comprehensive metrics for code quality and functional correctness
- **Reward Optimization**: GRPO implementation with execution feedback integration

## 🏆 Results & Impact

- **Improved Code Quality**: Significant enhancement in JAX-specific code generation accuracy
- **Functional Correctness**: High success rate in generated code execution and validation
- **Performance Optimization**: Efficient JAX primitive utilization in generated code
- **Scalable Framework**: Production-ready system with containerized deployment

## 🛠️ Technologies Used

- **Model**: Google CodeGemma-3
- **Framework**: JAX, FastAPI, Docker
- **ML Techniques**: Fill-in-the-Middle training, GRPO, Reinforcement Learning
- **Evaluation**: Custom metrics, execution validation, automated testing

