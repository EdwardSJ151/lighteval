from lighteval.metrics.metrics import Metrics

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.tasks.templates.utils.formatting_utils import capitalize, char_to_num

from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def enem_pfn(line, task_name: str = None):
    """Prompt function para o dataset do ENEM."""

    # Definindo algumas variaveis que são padrão para o Lighteval
    translation_literals = TRANSLATION_LITERALS[Language.PORTUGUESE]
    answer_word = capitalize(translation_literals.answer)

    # Definindo a instrução para o Lighteval
    instruction = "As perguntas a seguir são questões de múltipla escolha do Exame Nacional do Ensino Médio (ENEM), selecione a única alternativa correta e responda apenas com as letras \"A\", \"B\", \"C\", \"D\" ou \"E\".\n\n"

    description = ""
    if "description" in line and line["description"]:
        # Removendo listas vazias
        if isinstance(line["description"], list):
            if line["description"]:  # Apenas processa listas não vazias
                description = "\n".join(str(item) for item in line["description"])
        else:
            description = str(line["description"])

    # Format question
    question = line["question"]
    question = question.replace("[[placeholder]]", "").strip()

    # Cria query dependendo se a descrição existe ou não
    query = instruction
    if description:
        query += f"{description}\n\nPergunta: {question}\n"
    else:
        query += f"Pergunta: {question}\n"

    choices = [
        line["alternative_a"],
        line["alternative_b"],
        line["alternative_c"],
        line["alternative_d"],
        line["alternative_e"],
    ]

    valid_keys = ["A", "B", "C", "D", "E"]

    answer_index = line["label"]
    answer_index = [char_to_num(answer_index)] # Indice da resposta correta em formato de lista

    # Criar fim da query com as opções de resposta
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


enem_task = LightevalTaskConfig(
    name="enem",
    suite=["ptbench"],
    prompt_function=enem_pfn,
    hf_repo="EdwardSJ151/enem-lighteval-fewshot",
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