import json
import sys
from datetime import UTC, datetime

from loguru import logger

# Remove a configuração de logging padrão do Loguru para evitar duplicação de logs
logger.remove()


def configure_logging(message):
    """Configura o logging para enviar logs em formato JSON para stdout."""
    record = message.record
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": record["level"].name.lower(),
        "logger": record["extra"].get("logger_name", record["name"]),
        "event": record["message"],
    }
    print(json.dumps(payload, ensure_ascii=False), file=sys.stdout)


# Adiciona a configuração de logging ao logger do Loguru
logger.add(configure_logging, level="INFO")


def get_logger(name):
    """Retorna um logger configurado com o nome especificado."""
    return logger.bind(logger_name=name)
