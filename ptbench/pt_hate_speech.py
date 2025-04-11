from lighteval.metrics.metrics import Metrics

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.tasks.templates.utils.formatting_utils import capitalize

from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

def pt_hate_speech_pfn(line, task_name: str = None):
    """Prompt function para o dataset do PT Hate Speech."""

    # Definindo algumas variaveis que são padrão para o Lighteval
    translation_literals = TRANSLATION_LITERALS[Language.PORTUGUESE]
    answer_word = capitalize(translation_literals.answer)

    # Definindo a instrução para o Lighteval
    instruction = "Abaixo contém o texto de tweets de usuários do Twitter em português, sua tarefa é classificar se o texto contém discurso de ódio ou não. Selecione entre apenas os seguintes números: 1 (Hate Speech) e 0 (Não Hate Speech).\n\n"

    tweet = line["sentence"]
    query = instruction
    query += f"Texto: {tweet}\n\nPergunta: O texto contém discurso de ódio?\n"

    valid_keys = ["0", "1"]

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


pt_hate_speech_task = LightevalTaskConfig(
    name="pt_hate_speech",
    suite=["ptbench"],
    prompt_function=pt_hate_speech_pfn,
    hf_repo="EdwardSJ151/portuguese_hate_speech_lighteval_fewshot",
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

# TODO: Read up on why I cant use f1 score on multiple choice single token tasks

# metric=[Metrics.loglikelihood_acc_norm,
#         Metrics.exact_match,
#         Metrics.quasi_exact_match,
#         Metrics.prefix_exact_match,
#         Metrics.prefix_quasi_exact_match,
#         Metrics.f1_score_macro,
#         Metrics.f1_score_micro,
#         ],