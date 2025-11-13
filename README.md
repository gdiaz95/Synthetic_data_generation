## 🚀 Getting Started

### 1. (Required) Configure GPU Environment

Before you install anything, you **must** create a `.env` file in the root of this project. This file tells the program which GPU(s) to use.

Create a file named `.env` in the project root and add **one** of the following lines, depending on your hardware:

**To use a single, specific GPU (e.g., GPU 0):**

CUDA_VISIBLE_DEVICES=0

**To force CPU-only (if no GPU is available):**

CUDA_VISIBLE_DEVICES=