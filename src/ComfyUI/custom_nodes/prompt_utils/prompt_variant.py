class PromptVariant:
    def __init__(self, version: str, test: str, subject: str, style: str,
                 iteration: int = 1, seed: int = 0, cfg: float = 7.0, steps: int = 20):
        self.version = version
        self.test = test
        self.subject = subject
        self.style = style

        # Инфраструктурные параметры для Grid Search / Свипа параметров
        self.iteration = iteration
        self.seed = seed
        self.cfg = cfg
        self.steps = steps

    @property
    def full_prompt(self) -> str:
        subject = (self.subject or "").strip()
        style = (self.style or "").strip()

        parts = [p for p in [subject, style] if p]
        return ", ".join(parts)

    def __repr__(self):
        return f"PromptVariant(version={self.version}, test={self.test}, cfg={self.cfg}, steps={self.steps})"