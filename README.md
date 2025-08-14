# Align-CodeGemma: JAX-Aware Code Generation & Alignment

A comprehensive framework for fine-tuning Google's **Gemma-3** to generate high-quality **JAX-based code** for computational tasks. This project implements advanced alignment techniques, execution-time validation, and reward-driven optimization using **GRPO** (Guided Reinforcement with Prompt Objectives).

## 🎯 Project Overview

This project addresses the challenge of generating domain-specific JAX code by:
- Fine-tuning Gemma-3 4B with specialized alignment techniques
- Implementing real-time code execution validation
- Using reinforcement learning to optimize code quality
- Developing custom metrics for JAX primitive evaluation

## 🚀 Key Features

- **GRPO Optimization**: Custom reinforcement learning approach for code generation improvement
- **Execution Validation**: Real-time code testing and validation pipeline
- **JAX-Specific Metrics**: Custom evaluation framework for JAX primitive usage
- **Scalable Architecture**: Docker-based deployment with FastAPI execution server

## 📁 Project Structure

```
Align-CodeGemma/
├── datas/                          
│   |── train_meta.json           
│      
│
├── execserver/                    # Code execution and validation system
│   ├── Dockerfile                 # docker file 
│   ├── build-run.sh               # shell file
│   └── code_exec_reqs.py          # server endpoints
│
├── src/                         
│   ├── Docker image               # Docker image with pytorch and deepspeed
│   ├── deepspeed.yaml             # deepspeed configuration
│   ├── GRPO.yaml                  # grpo configuration
│   └── prompt_template.py         # prompt template - train
|   |__ train.py                   
|   |__ utils.py                                       
├── .gitattributes
└── README.md
```

## 🔧 Technical Implementation

### Data Processing (`datas/`)
- **Data Preparation**: Natural language descriptions of high compute intensive tasks

### Execution Server (`execserver/`)
- **Validation Pipeline**: Rust code execution with security sandboxing
- **API Framework**: RESTful endpoints for code testing and result aggregation
- **JAX Analysis**: Automated detection and validation of JAX primitives using regex

## 🏆 Results & Impact

- **Improved Code Quality**: Significant enhancement in JAX-specific code generation accuracy
- **Functional Correctness**: High success rate in generated code execution and validation
- **Performance Optimization**: Efficient JAX primitive utilization in generated code
- **Scalable Framework**: Production-ready system with containerized deployment

## 🛠️ Technologies Used

- **Model**: Google CodeGemma-3
- **Framework**: JAX, RESTAPI, Docker
- **ML Techniques**:  GRPO
- **Evaluation**: Custom reward metrics, execution validation.

