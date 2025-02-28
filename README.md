# MLOps-Deployment

## Overview
This repository documents the deployment and serving of Machine Learning (ML) models, Deep Learning models, and Large Language Models (LLMs) in production. It covers the full ML pipeline, model optimization, and serving strategies using various tools and frameworks.

## Features
- **ML Pipeline:** DVC, MLflow, GitHub Actions, deploying models as API endpoints on VMs
- **Model Optimization & Serving:** TensorRT + Triton for common ML and Deep Learning models
- **Model Serving:**
  - General ML models (Scikit-learn, XGBoost, etc.)
  - Deep Learning models (TensorFlow, PyTorch)
  - Large Language Models (LLMs):
    - Llama.cpp for GGUF format
    - Model conversion to GGUF
    - vLLM for efficient LLM inference
    - TensorRT-LLM for optimized LLM serving

## Required Knowledge
To effectively use this repository, familiarity with the following concepts and tools is recommended:
- Machine Learning model lifecycle and MLOps principles
- Docker & Kubernetes for containerized deployment
- FastAPI for building and serving API endpoints
- GitHub Actions for CI/CD automation
- DVC & MLflow for model tracking and versioning
- TensorRT & Triton for model optimization and inference
- ML, Deep Learning, and LLM frameworks such as TensorFlow, PyTorch, Llama.cpp, vLLM, and TensorRT-LLM

## Progress Tracking
A status table tracks progress on different components. It is automatically updated via a GitHub Action when tasks are completed.

| Task | Status |
|------|--------|
| Setup DVC & MLflow | ❌ |
| Implement GitHub Actions | ❌ |
| Deploy ML model as API | ❌ |
| Deploy Deep Learning model as API | ❌ |
| Deploy LLM as API | ❌ |
| Convert model to GGUF | ❌ |
| Setup Llama.cpp | ❌ |
| Deploy vLLM | ❌ |
| Optimize with TensorRT-LLM | ❌ |

## Automating Status Updates
A GitHub Action updates this README when a task is completed. The workflow checks for commits referencing task completion and modifies the table accordingly.

### Example Workflow (GitHub Action)
```yaml
name: Update Task Status
on:
  push:
    branches:
      - main
jobs:
  update-status:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Update README
        run: |
          sed -i 's/| Setup DVC & MLflow | ❌ |/| Setup DVC & MLflow | ✅ |/' README.md

      - name: Commit and push changes
        run: |
          git config --global user.name 'github-actions'
          git config --global user.email 'github-actions@github.com'
          git add README.md
          git commit -m "Update task status"
          git push
```

## Usage
1. Clone the repository: `git clone https://github.com/your-username/MLOps-Deployment.git`
2. Follow documentation in each module for setup and deployment.
3. Contribute by improving scripts and adding new serving techniques.

## Contributing
Pull requests are welcome! Feel free to add enhancements and optimizations.

## License
MIT License

