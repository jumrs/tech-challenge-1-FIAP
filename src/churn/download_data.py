import shutil
from pathlib import Path
import kagglehub

from churn.logger import get_logger


logger = get_logger(__name__)

DATASET_NAME = "jpacse/datasets-for-churn-telecom"
OUTPUT = Path("data/raw/churn.csv")
FILENAME = 'cell2celltrain.csv'


def download_churn_dataset(
    dataset: str = DATASET_NAME,
    output: str | Path = OUTPUT) -> Path:
    """ Baixa o dataset de churn da telecom e salva no caminho especificado. """

    # Baixa o dataset do KaggleHub e move o arquivo CSV para o caminho de saída.
    dataset_path = Path(kagglehub.dataset_download(dataset, force_download=True))
    source_file = dataset_path / FILENAME
    if not source_file.exists():
        logger.error(f"Arquivo {FILENAME} não encontrado no dataset {dataset}.")
        raise FileNotFoundError(f"Arquivo {FILENAME} não encontrado no dataset {dataset}.")

    # Cria o diretório de saída, se não existir, e move o arquivo CSV para lá.
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, output_path)  
    logger.info(f"Arquivo {FILENAME} movido para {output_path}.")
    return output_path

def main():
    """ Função principal para baixar o dataset de churn. """
    try:
        download_churn_dataset()
        logger.info("Download do dataset de churn concluído com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao baixar o dataset de churn: {e}")
        raise


if __name__ == "__main__":
    main()
