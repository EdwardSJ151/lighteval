from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import (
    LogProbCharNorm,
    LogProbTokenNorm,
)

from lighteval.metrics.metrics import Metric, MetricCategory, Metrics
from lighteval.metrics.utils.metric_utils import MetricUseCase

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.utils.formatting_utils import capitalize, char_to_num
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

ENEM_SUBSETS = ["2022", "2023", "2024"]

def enem_pfn(line, task_name: str = None):
    """Prompt function for ENEM dataset."""
    translation_literals = TRANSLATION_LITERALS[Language.PORTUGUESE]
    answer_word = capitalize(translation_literals.answer)
    instruction = ""
    
    description = ""
    if "description" in line and line["description"]:
        # Removing empty list brackets
        if isinstance(line["description"], list):
            if line["description"]:  # Only process non-empty lists
                description = "\n".join(str(item) for item in line["description"])
        else:
            description = str(line["description"])
    
    # Format question
    question = line["question"]
    question = question.replace("[[placeholder]]", "").strip()
    if description:
        query = f"{description}\n\nPergunta: {question}\n"
    else:
        query = f"Pergunta: {question}\n"
    
    choices = [
        line["alternative_a"],
        line["alternative_b"],
        line["alternative_c"],
        line["alternative_d"],
        line["alternative_e"],
    ]
    
    valid_keys = ["A", "B", "C", "D", "E"]
    
    answer_index = line["label"]
    answer_index = [char_to_num(answer_index)] # Make to a list
    
    # Build the query with question and options
    options_text = "\n".join([f"{key}. {choice}" for key, choice in zip(valid_keys, choices)])
    query = f"{query}{options_text}\n"

    
    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys,
        gold_index=answer_index,
        instruction=instruction,
        unconditioned_query=f"{answer_word}{translation_literals.colon}",
    )

def get_enem_prompt_function():
    """Returns a function that generates prompts for ENEM multiple choice questions."""
    
    def format_description(desc_list):
        """Process description list into formatted string or empty string if list is empty"""
        if not desc_list:  # Handle empty list case
            return ""
        if isinstance(desc_list, list):
            return "\n".join(str(item) for item in desc_list)
        return str(desc_list)
    
    def format_text(text):
        """For handling lists to be formatted as a string"""
        if isinstance(text, list):
            return "\n".join(str(item) for item in text)
        return str(text)
    
    return get_mcq_prompt_function(
        language=Language.PORTUGUESE,
        adapter=lambda line: {
            "question": (format_description(line.get("description", "")) + "\n\n" + format_text(line["question"])).strip() 
                         if line.get("description") and line["description"] else format_text(line["question"]),
            "choices": [
                format_text(line["alternative_a"]),
                format_text(line["alternative_b"]),
                format_text(line["alternative_c"]),
                format_text(line["alternative_d"]),
                format_text(line["alternative_e"]),
            ],
            "gold_idx": char_to_num(line["label"]),
            "instruction": ""
        },
        formulation=MCFFormulation()
    )


class ENEMTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            # prompt_function=enem_pfn,
            prompt_function=get_enem_prompt_function(),
            hf_repo="EdwardSJ151/enem-lighteval", # "MBZUAI/ArabicMMLU",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["ptbench"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            version=0,
        )


ENEM_TASKS = [
    ENEMTask(name=f"ENEM:{subset}", hf_subset=subset) for subset in ENEM_SUBSETS
]