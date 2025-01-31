# Evaluating LLM Reasoning in the Operations Research Domain with ORQA

This repo contains the demo and the implementation of the paper 'Evaluating LLM Reasoning in the Operations Research Domain with ORQA', published at AAAI2025.

Link to arXiv paper: https://arxiv.org/abs/2412.17874
Link to Huawei's AI Gallery Notebook: [https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=58b799a0-5cfc-4c2e-8b9b-440bb2315264](https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=6b98c56e-913b-47ef-8d9f-3266c8aec06a)

Operations Research Question Answering (ORQA) is a new benchmark designed to assess the reasoning capabilities of Large Language Models (LLMs) in a specialized technical domain, namely Operations Research (OR). The benchmark evaluates whether LLMs can emulate the knowledge and reasoning skills of OR experts when presented with complex optimization problems. Crafted by OR experts, the dataset consists of real-world optimization problems that require multi-step mathematical reasoning to arrive at solutions. Our evaluations of several open-source LLMs, such as LLaMA 3.1, DeepSeek, and Mixtral, reveal their modest performance, highlighting a gap in their ability to generalize to specialized technical domains.


ORQA questions are hand-crafted to require complex, multi-step reasoning to identify the components of mathematical models and their interrelationships. An example of these components and their corresponding mathematical formulations is shown below.

<p align="center">
  <img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/orqa/img/ORQA-Fig2.png" width="1000" />
</p>


---
## Download the dataset
```python
!wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/orqa/dataset.zip
!unzip -qo dataset.zip
```

---
## Environment Setup

### Step 1: Create and Activate the Conda Environment

First, create and activate a conda environment with Python 3.11.4:

```bash
conda create --name orqa_py3.11 python=3.11.4
conda activate orqa_py3.11
```

### Step 2: Install Dependencies

If your device uses **CUDA Version 12.2**, you can install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If you have a different CUDA version or if the above command doesn't work, you can install the necessary packages individually:

```bash
pip install huggingface_hub tenacity evaluate
```

---

## Dataset Overview

The dataset can be found in the directory: `src/task/dataset`.

It includes two files:

- **Test Set (1468 instances)**: `ORQA_test.jsonl`
- **Validation Set (45 instances)**: `ORQA_validation.jsonl`

### Each Dataset Instance Contains:

1. **CONTEXT**: A description of an optimization problem presented as a case study in natural language.

2. **QUESTION**: A question related to the problem's specifications, underlying model components, or the logic of the optimization model. It might ask about:
   - Objective criteria or constraints
   - Model components (e.g., elements in the optimization)
   - Relationships between components

3. **OPTIONS**: A list of four possible answers. These are created by OR experts to make the question challenging. The LLM must choose the correct answer from these options.

4. **TARGET_ANSWER**: The correct answer to the question.

5. **REASONING**: For validation set only, which contains expert-created reasoning steps that explain how the correct answer is derived.

<p align="center">
  <img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/orqa/img/data_breakdown.PNG" width="1000" />
</p>

### Example Instance (Validation Set)

Below is an example instance from the validation split, which includes expert-created reasoning steps used for in-context learning. **Note**: The test set instances do not contain these reasoning steps.

```json
instance = {
  "QUESTION_TYPE": "Q6", 
  "CONTEXT": "As a programming director at the Starlight Network, you're tasked with creating a lineup for the prime-time broadcasting...",
  "QUESTION": " What are the decision activities of the optimization problem?",
  "OPTIONS": ["Due date", "Show broadcast order", "Show broadcast indicator", "Processing time"], 
  "ARGET_ANSWER": 2, 
  "REASONING": "The possible decision activities mentioned in options ..."
}
```
---


## Building the Prompt (Implemented in the Code)

The **prompt** is constructed using specific keys from the dataset. Below is how prompts are buildin our experiemnts.

### **Standard (0-shot) Prompting**

The prompt is built in `/src/task/base_task.py` and `/src/task/standard_task.py`. The format for the prompt is as follows:

```python
Given the context (following Context:), select the most appropriate answer to the question (following Question:). Answer only 'A', 'B', 'C', or 'D'
Context: {instance["CONTEXT"]}
Question: {instance["question"]}
A. {instance["OPTIONS"][0]}
B. {instance["OPTIONS"][1]}
C. {instance["OPTIONS"][2]}
D. {instance["OPTIONS"][3]}
Answer: Among A through D, the answer is (
```

### **CoT (0-shot) Prompting**

The prompts are built in `/src/task/base_task.py` and `/src/task/cot_task.py`.

<u>First (Reasoning Eliciting) Prompt:</u>

```python
Given the context (following Context:), provide the chain of thoughts (following Reasoning:) to solve the question (following Question:). Remember, only one option is correct.
Context: {instance["CONTEXT"]}
Question: {instance["question"]}
A. {instance["OPTIONS"][0]}
B. {instance["OPTIONS"][1]}
C. {instance["OPTIONS"][2]}
D. {instance["OPTIONS"][3]}
Reasoning: {os.environ['trigger']}
```

- The **trigger prompt** is by default set to `"Let's think step by step"`.

<u>Second (answering) prompt:</u>

Let the output reasoning of the reasoning eliciting prompt be `REASONING`.

```python
Given the context (following Context:), the reasoning (following Reasoning:), select the most appropriate answer to the question (following Question:). Answer only 'A', 'B', 'C', or 'D'. There is only one correct answer.
Context: {instance["CONTEXT"]}
Question: {instance["QUESTION"]}
A. {instance["OPTIONS"][0]}
B. {instance["OPTIONS"][1]}
C. {instance["OPTIONS"][2]}
D. {instance["OPTIONS"][3]}
Reasoning: {os.environ['trigger']}. {REASONING}
Answer: Among A through D, the answer is (
``` 
---

## Running Experiments

Before running the experiments, ensure you pass your Huggingface API token either as an argument (`--token hf_xxxxxx`) or set it as a default in `standard_inference.py` and/or `cot_inference.py`.

- `cot_inference.py`: Runs COT experiments
- `standard_inference.py`: Runs standard n-shot experiments
- Both scripts use `final_inference_utils.py` for global variables and utility functions, including model names.

### Running the Experiments

To generate the main accuracy results for each model, run the following scripts:

```bash
bash src/standard_0-shot.sh
bash src/standard_1-shot.sh
bash src/standard_3-shot.sh
bash src/cot_0-shot.sh
bash src/cot_1-shot.sh
```

The table below shows the accuracy of different models across various prompting strategies.
<p align="center">
  <img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/orqa/img/table_2.PNG" width="500" />
</p>

### Trigger Prompt Analysis

To analyze the impact of different trigger prompts on accuracy, run the following script:

```bash
bash src/trigger_prompt_analysis.sh
```

The results for the Trigger Prompt are averaged over five independent runs per prompt. The final row displays the accuracy of ensembling all prompts.
<p align="center">
  <img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/orqa/img/table_3.PNG" width="700" />
</p>

### Results Storage

All results will be saved in `/src/output_of_llms`.
