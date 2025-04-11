from lighteval.tasks.lighteval_task import LightevalTaskConfig
from ptbench.enem import enem_task
from ptbench.pt_hate_speech import pt_hate_speech_task

TASKS_TABLE: list[LightevalTaskConfig] = (
    [enem_task] +
    [pt_hate_speech_task]
)
