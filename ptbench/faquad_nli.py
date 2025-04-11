from lighteval.metrics.metrics import Metrics

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.tasks.templates.utils.formatting_utils import capitalize

from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

def faquad_nli_pfn(line, task_name: str = None):
    return NotImplemented

faquad_nli_task = LightevalTaskConfig(
    name="faquad_nli",
    suite=["ptbench"],
    prompt_function=faquad_nli_pfn,
    hf_repo="EdwardSJ151/faquad_nli_lighteval_fewshot",
    hf_subset="default",
    hf_avail_splits=["test", "dev"],
    evaluation_splits=["test"],
    few_shots_split="dev",
    few_shots_select="sequential",
    generation_size=-1,
    metric=[Metrics.f1_score_macro],
    stop_sequence=None,
    trust_dataset=True,
    version=0,
)