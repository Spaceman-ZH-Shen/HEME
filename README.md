# Knowledge Enhancement with Multi-step Ensemble

## Introduction

We present HEME, framework that advances multimodal reasoning capabilities through systematic context enrichment and multi-stage inference. Our approach addresses the challenge of reasoning about complex visual-linguistic relationships by decomposing the reasoning process into three hierarchical stages: perceptual grounding, contextual reasoning, and ensemble selection.  leveraging the reasoning capabilities of large language models (LLMs) to understand spatial relationships, scene semantics, and object interactions, our framework achieves 61.9% ACC@0.5 on the LENS benchmark. 

---

## Model Famework
![](FIG/Famework.jpg)


---

## Included Models
The project contains the following pretrained models. You can place the downloaded model files under the directory `SUP_model_project/basic_model`, or alternatively, specify the model paths in the `config.py` file.

- **Qwen2.5-VL-72B-Instruct (https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)**  
- **Qwen2.5-VL-32B-Instruct (https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)**  
- **Florence-2-large (https://huggingface.co/microsoft/Florence-2-large)**  
- **Florence-2-large-ft (https://huggingface.co/microsoft/Florence-2-large-ft)**  
- **grounding-dino-base (https://huggingface.co/IDEA-Research/grounding-dino-base)**  
- **llmdet_swin_large_hf (https://huggingface.co/fushh7/llmdet_swin_large_hf)**

---

## Environment Setup
  
Before running any code, set the environment variable to define the project path:  
```
export PYTHONPATH=/path/to/SUP_model_project
```
Replace /path/to/SUP_model_project with the absolute path on your system.
2. **Transformers Version Compatibility** 

The Florence, Grounding DINO, and llmdet models require transformers version 4.41.0.

This version may cause errors when running Qwen2.5-VL models. If you encounter errors, upgrade transformers to 4.52.4 when running Qwen-related code:

```
pip install transformers==4.52.4
```
It is recommended to use separate Python virtual environments for different models to avoid version conflicts.

---

## Execution Instructions  

Run scripts in the order indicated by their numeric prefixes, for example:

```
1.1_XXX.py -> 1.2_XXX.py -> 1.3_XXX.py -> 2.1_XXX.py -> 2.2_XXX.py ...
```
If GPU memory allows, scripts with the same prefix (e.g., multiple 1.1_ scripts) can be run in parallel to speed up processing.

Refer to script headers and comments for detailed descriptions of each step.

When running the file step1.1_val_llmdet, you may encounter a problem where the code package cannot be found. If an error occurs, you can run the file under the official code package of llmdet (https://github.com/iSEE-Laboratory/LLMDet/tree/main).

Project Structure Example
```
SUP_model_project/
│
├── basic_model/                
│   ├── qwen2.5-vl-72b-instruct/
│   ├── qwen2.5-vl-32b-instruct/
│   ├── florence-2-large/
│   ├── grounding-dino-base/
│   └── llmdet_swin_large_hf/
├── knowledge
├── img_size_transfer_tool
├── baseline/                   
│   ├── baseline_results
│   ├── images
│   ├── Qwen_baseline/
│   │   ├── img_1120
│   │   │   ├── images
│   │   │   ├── step1.3_...py
│   │   │   └── ...
│   │   └── img_1400
│   │       ├── images
│   │       ├── step1.3_...py
│   │       └── ...
│   ├── 1.1_...py
│   ├── 1.2_...py
│   └── ...
├── step2/                     
│   ├── 2.1_...py
│   ├── 2.2_...py
│   └── ...
└── step3/                     
    ├── 3.1_...py
    ├── 3.2_...py
    └── ...
```
All experiments were run on 4 NVIDIA H200 GPUs. 

## Team Members
Yashu Kang - Zhejiang Supcon Information Technology & Zhejiang University of Technology - kangyashu@supconit.com

Zhehao Shen - Zhejiang Supcon Information Technology & Soochow University - [@Spaceman-ZH-Shen](https://github.com/Spaceman-ZH-Shen) - 20234246028@stu.suda.edu.cn

Yuzhe Cen - Zhejiang Supcon Information Technology & Columbia University -[@MiCENzz](https://github.com/MiCENzz) - yc4494@columbia.edu





