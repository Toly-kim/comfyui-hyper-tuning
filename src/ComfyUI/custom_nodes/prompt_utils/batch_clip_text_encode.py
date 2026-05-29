import torch

class BatchCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_list": ("LIST",),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "custom/orchestration"

    def encode(self, text_list, clip):
        cond_list = []
        pooled_list = []

        for text in text_list:
            # Предотвращаем падение на пустых строках
            clean_text = str(text).strip() if text else "score baseline"
            tokens = clip.tokenize(clean_text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)
            pooled_list.append(pooled)

        # Склеиваем по батчевой оси для параллельного процессинга
        batched_cond = torch.cat(cond_list, dim=0)
        batched_pooled = torch.cat(pooled_list, dim=0)

        return ([[batched_cond, {"pooled_output": batched_pooled}]],)
