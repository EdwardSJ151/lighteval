from lighteval.tasks.lighteval_task import LightevalTaskConfig
from ptbench.enem import enem_task
from ptbench.pt_hate_speech import pt_hate_speech_task
from ptbench.faquad_nli import faquad_nli_task
from ptbench.assin2rte import assin2rte_task

TASKS_TABLE: list[LightevalTaskConfig] = (
    [enem_task] +
    [pt_hate_speech_task] +
    [faquad_nli_task] +
    [assin2rte_task]
)
