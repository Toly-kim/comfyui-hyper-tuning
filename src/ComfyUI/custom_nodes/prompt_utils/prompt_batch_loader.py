import json
from .prompt_variant import PromptVariant

class PromptBatchLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "variants_json": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }

    # Расширяем выходной интерфейс ComfyUI новыми списками гиперпараметров
    RETURN_TYPES = ("LIST", "LIST", "LIST", "LIST", "LIST", "INT", "LIST", "LIST", "LIST", "LIST")
    RETURN_NAMES = (
        "full_prompts",
        "subjects",
        "styles",
        "versions",
        "tests",
        "batch_size",
        "iteration_list",
        "seed_list",
        "cfg_list",      # Передаем напрямую в MicroBatchKSampler
        "steps_list"     # Передаем напрямую в MicroBatchKSampler
    )
    FUNCTION = "load_batch"
    CATEGORY = "custom/orchestration"
    OUTPUT_NODE = True

    def load_batch(self, variants_json):
        try:
            data = json.loads(variants_json)
            prompts = []
            iterations = []
            seeds = []
            cfgs = []
            steps_out = []

            for item in data:
                # Парсим базовый текстовый вариант
                p = PromptVariant(
                    version=str(item.get("version", "v?")),
                    test=item.get("test", "none"),
                    subject=item.get("subject", ""),
                    style=item.get("style", "")
                )

                # Заполняем метаданные эксперимента с явным приведением типов
                p.iteration = int(item.get("iteration", 1))
                p.seed = int(item.get("seed", 0))
                p.cfg = float(item.get("cfg", 7.0))
                p.steps = int(item.get("steps", 20))

                prompts.append(p)
                iterations.append(p.iteration)
                seeds.append(p.seed)
                cfgs.append(p.cfg)
                steps_out.append(p.steps)

            full_prompts = [p.full_prompt for p in prompts]
            subjects = [p.subject for p in prompts]
            styles = [p.style for p in prompts]
            versions = [p.version for p in prompts]
            tests = [p.test for p in prompts]
            batch_size = len(prompts)

            print(f"📦 [LOADER] Successfully unrolled matrix data of {batch_size} items for Hyperparameter Tuning.")

            return (full_prompts, subjects, styles, versions, tests, batch_size, iterations, seeds, cfgs, steps_out)

        except Exception as e:
            print(f"❌ [LOADER] Critical crash parsing variants JSON: {e}")
            # Безопасный дефолтный фоллбэк, чтобы граф ComfyUI не падал при опечатках в JSON
            return ([], [], [], [], [], 0, [], [], [], [])