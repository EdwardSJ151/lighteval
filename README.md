# Lighteval Fork for Portuguese Benchmark

## Setup
`pip install -e .[adapters,math,dev]`

or for a full install

`pip install -e .[quantization,vllm,tensorboardX,adapters,math,dev]`

## Evaluation Tasks Definitions:
Tasks are defined in the following format `pt_benchmark|{task_name}|{few_shot}|{truncate_few_shots}`

## Creating a Task:

1. Implement the task as a new file in the `pt_benchmark/` dir. Follow the existing lighteval task implementation as reference.
2. Add the task in the `TASK_TABLE` constant in the `community_tasks/pt_evals.py` file.
3. Ensure that nothing is amiss&mdash; inspect the task using `python -m lighteval tasks inspect` to examine a single sample.
E.g. `python -m lighteval tasks inspect "ptbench|enem|3|0"   --num-samples 1   --custom-tasks community_tasks/pt_evals.py`
4. If everything looks good, add the task string, i.e., `ptbench|{task_name}|{few_shot}|{truncate_few_shots}` in the `examples/tasks/all_ptbench_tasks.txt` file.

# Running a Task on a Model:
```
python -m lighteval accelerate \
    "pretrained=TucanoBR/Tucano-1b1" \
    "ptbench|pt_hate_speech|25|1" \
    --custom-tasks community_tasks/pt_evals.py \
    --save-details
```

<p align="center">
  <br/>
    <img alt="lighteval library logo" src="./assets/lighteval-doc.svg" width="376" height="59" style="max-width: 100%;">
  <br/>
</p>


<p align="center">
    <i>Your go-to toolkit for lightning-fast, flexible LLM evaluation, from Hugging Face's Leaderboard and Evals Team.</i>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/tests.yaml?query=branch%3Amain)
[![Quality](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml/badge.svg?branch=main)](https://github.com/huggingface/lighteval/actions/workflows/quality.yaml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lighteval)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/huggingface/lighteval/blob/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/lighteval)](https://pypi.org/project/lighteval/)

</div>

---

**Documentation**: <a href="https://huggingface.co/docs/lighteval/index" target="_blank">Lighteval's Wiki</a>

---

### Unlock the Power of LLM Evaluation with Lighteval 🚀

**Lighteval** is your all-in-one toolkit for evaluating LLMs across multiple
backends—whether it's
[transformers](https://github.com/huggingface/transformers),
[tgi](https://github.com/huggingface/text-generation-inference),
[vllm](https://github.com/vllm-project/vllm), or
[nanotron](https://github.com/huggingface/nanotron)—with
ease. Dive deep into your model’s performance by saving and exploring detailed,
sample-by-sample results to debug and see how your models stack-up.

Customization at your fingertips: letting you either browse all our existing [tasks](https://huggingface.co/docs/lighteval/available-tasks) and [metrics](https://huggingface.co/docs/lighteval/metric-list) or effortlessly create your own [custom task](https://huggingface.co/docs/lighteval/adding-a-custom-task) and [custom metric](https://huggingface.co/docs/lighteval/adding-a-new-metric), tailored to your needs.

Seamlessly experiment, benchmark, and store your results on the Hugging Face
Hub, S3, or locally.


## 🔑 Key Features

- **Speed**: [Use vllm as backend for fast evals](https://huggingface.co/docs/lighteval/use-vllm-as-backend).
- **Completeness**: [Use the accelerate backend to launch any models hosted on Hugging Face](https://huggingface.co/docs/lighteval/quicktour#accelerate).
- **Seamless Storage**: [Save results in S3 or Hugging Face Datasets](https://huggingface.co/docs/lighteval/saving-and-reading-results).
- **Python API**: [Simple integration with the Python API](https://huggingface.co/docs/lighteval/using-the-python-api).
- **Custom Tasks**: [Easily add custom tasks](https://huggingface.co/docs/lighteval/adding-a-custom-task).
- **Versatility**: Tons of [metrics](https://huggingface.co/docs/lighteval/metric-list) and [tasks](https://huggingface.co/docs/lighteval/available-tasks) ready to go.


## ⚡️ Installation

```bash
pip install lighteval
```

Lighteval allows for many extras when installing, see [here](https://huggingface.co/docs/lighteval/installation) for a complete list.

If you want to push results to the Hugging Face Hub, add your access token as
an environment variable:

```shell
huggingface-cli login
```

## 🚀 Quickstart

Lighteval offers the following entry points for model evaluation:

- `lighteval accelerate` : evaluate models on CPU or one or more GPUs using [🤗
  Accelerate](https://github.com/huggingface/accelerate)
- `lighteval nanotron`: evaluate models in distributed settings using [⚡️
  Nanotron](https://github.com/huggingface/nanotron)
- `lighteval vllm`: evaluate models on one or more GPUs using [🚀
  VLLM](https://github.com/vllm-project/vllm)
- `lighteval endpoint`
    - `inference-endpoint`: evaluate models on one or more GPUs using [🔗
  Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated)
    - `tgi`: evaluate models on one or more GPUs using [🔗 Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/index)
    - `openai`: evaluate models on one or more GPUs using [🔗 OpenAI API](https://platform.openai.com/)

Here’s a quick command to evaluate using the Accelerate backend:

```shell
lighteval accelerate \
    "pretrained=gpt2" \
    "leaderboard|truthfulqa:mc|0|0"
```

## 🙏 Acknowledgements

Lighteval started as an extension of the fantastic [Eleuther AI
Harness](https://github.com/EleutherAI/lm-evaluation-harness) (which powers the
[Open LLM
Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard))
and draws inspiration from the amazing
[HELM](https://crfm.stanford.edu/helm/latest/) framework.

While evolving Lighteval into its own standalone tool, we are grateful to the
Harness and HELM teams for their pioneering work on LLM evaluations.

## 🌟 Contributions Welcome 💙💚💛💜🧡

Got ideas? Found a bug? Want to add a
[task](https://huggingface.co/docs/lighteval/adding-a-custom-task) or
[metric](https://huggingface.co/docs/lighteval/adding-a-new-metric)?
Contributions are warmly welcomed!

If you're adding a new feature, please open an issue first.

If you open a PR, don't forget to run the styling!

```bash
pip install -e .[dev]
pre-commit install
pre-commit run --all-files
```
## 📜 Citation

```bibtex
@misc{lighteval,
  author = {Fourrier, Clémentine and Habib, Nathan and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.8.0},
  url = {https://github.com/huggingface/lighteval}
}
```
