from lighteval.tasks.lighteval_task import LightevalTaskConfig
from ptbench.enem import ENEM_TASKS
from ptbench.enem_shots import enem_task
TASKS_TABLE: list[LightevalTaskConfig] = (
    ENEM_TASKS
    + [enem_task]
)
