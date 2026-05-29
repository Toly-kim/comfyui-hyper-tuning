from pathlib import Path
import yaml

CONFIG_YAML = "config.yaml"
ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"
SRC_DIR = ROOT / "src"
EXPERIMENTS_DIR = ROOT / "experiments"
REPORTS_DIR = ROOT / "reports"
# DATA_DIR = ROOT / "data"
# WORKFLOWS = CONFIG / "workflows"

class PathManager:
    def __init__(self):
        self._ensure_dirs()

    def _ensure_dirs(self):
        for folder in [CONFIG_DIR, EXPERIMENTS_DIR, REPORTS_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

    def get_config_path(self, filename: str = CONFIG_YAML) -> Path:
        return CONFIG_DIR / filename

    def get_cfg(self):
        return self.load_yaml_config(CONFIG_DIR / CONFIG_YAML)

    # @classmethod
    # def get_config_path(cls, filename="config.yaml") -> str:
    #     """Возвращает абсолютный путь к файлу конфига."""
    #     path = cls.CONFIG / filename
    #     return str(path)

    def get_workflow_path(self, filename: str) -> Path:
        path = CONFIG_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Workflow JSON not found: {path}")
        return path

    @classmethod
    def get_experiment_dir(cls, experiment_name: str) -> Path:
        """Создает и возвращает путь к папке эксперимента (статичный доступ)."""
        path = EXPERIMENTS_DIR / experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # @classmethod
    # def get_data_path(cls, subfolder: str) -> Path:
    #     """Возвращает путь к папке с данными (baseline или advanced)."""
    #     return cls.DATA / subfolder

    @staticmethod
    def load_yaml_config(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

# Создаем глобальный экземпляр для удобного импорта
paths = PathManager()