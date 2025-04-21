from lighteval.metrics.metrics import Metrics

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.tasks.templates.utils.formatting_utils import capitalize

from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

def assin2rte_pfn(line, task_name: str = None):
    """Prompt function para o dataset do assin2 rte."""

    # Definindo algumas variaveis que são padrão para o Lighteval
    translation_literals = TRANSLATION_LITERALS[Language.PORTUGUESE]
    answer_word = capitalize(translation_literals.answer)

    # Definindo a instrução para o Lighteval
    instruction = "Abaixo estão pares de pergunta e resposta. Para cada par, você deve julgar se a resposta responde à pergunta de maneira satisfatória e aparenta estar correta. Responda com apenas 'Sim' (Pode ser inferida) ou 'Não' (Não pode ser inferida).\n\n"

    premise = line["sentence1"]
    hypothesis = line["sentence2"]

    query = instruction
    query += f"Premissa: {premise}\n\Hipótese: {hypothesis}\n\nPergunta: Pergunta: A hipótese pode ser inferida pela premissa?\n"

    valid_keys = ["Não", "Sim"]

    answer_index = line["label"]
    answer_index = [answer_index] # Indice da resposta correta em formato de lista

    query = f"{query}\n"


    return Doc(
        task_name=task_name,
        query=query,
        choices=valid_keys,
        gold_index=answer_index,
        instruction=instruction,
        unconditioned_query=f"{answer_word}{translation_literals.colon}",
    )

assin2rte_task = LightevalTaskConfig(
    name="assin2rte",
    suite=["ptbench"],
    prompt_function=assin2rte_pfn,
    hf_repo="EdwardSJ151/assin2rte_lighteval_fewshot",
    hf_subset="default",
    hf_avail_splits=["test", "dev"],
    evaluation_splits=["test"],
    few_shots_split="dev",
    few_shots_select="sequential",
    generation_size=-1,
    metric=[Metrics.loglikelihood_acc_norm],
    stop_sequence=None,
    trust_dataset=True,
    version=0,
)