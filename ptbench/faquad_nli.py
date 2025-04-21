from lighteval.metrics.metrics import Metrics

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.tasks.templates.utils.formatting_utils import capitalize

from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

def faquad_nli_pfn(line, task_name: str = None):
    """Prompt function para o dataset do faquad nli."""

    # Definindo algumas variaveis que são padrão para o Lighteval
    translation_literals = TRANSLATION_LITERALS[Language.PORTUGUESE]
    answer_word = capitalize(translation_literals.answer)

    # Definindo a instrução para o Lighteval
    instruction = "Abaixo estão pares de pergunta e resposta. Para cada par, você deve julgar se a resposta responde à pergunta de maneira satisfatória e aparenta estar correta. Responda com apenas 'Sim' (Satisfaz) ou 'Não' (Não Satisfaz).\n\n"

    pergunta = line["question"]
    resposta = line["answer"]

    query = instruction
    query += f"Pergunta: {pergunta}\n\nResposta: {resposta}\n\nA resposta dada satisfaz à pergunta?\n"

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
    metric=[Metrics.loglikelihood_acc_norm],
    stop_sequence=None,
    trust_dataset=True,
    version=0,
)