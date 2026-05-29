from .prompt_batch_loader import PromptBatchLoader
from .batch_clip_text_encode import BatchCLIPTextEncode
from .multi_metric_scorer import MultiMetricScorer
from .micro_batch_sampler import MicroBatchKSampler

NODE_CLASS_MAPPINGS = {
    "PromptBatchLoader": PromptBatchLoader,
    "BatchCLIPTextEncode": BatchCLIPTextEncode,
    "MultiMetricScorer": MultiMetricScorer,
    "MicroBatchKSampler": MicroBatchKSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptBatchLoader": "Prompt Batch Loader",
    "BatchCLIPTextEncode": "Batch CLIP Text Encode",
    "MultiMetricScorer": "Multi Metric Scorer",
    "MicroBatchKSampler": "Micro Batch KSampler"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
