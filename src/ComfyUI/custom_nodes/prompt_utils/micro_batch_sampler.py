import nodes
import torch
import comfy

class MicroBatchKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_micro_batch_size": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "seed_list": ("LIST",),
                "cfg_list": ("LIST",),
                "steps_list": ("LIST",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "custom/sampling"

    def slice_conditioning(self, conditioning, start, end):
        """Безопасно нарезает кондишенинг. Если это свип параметров (текст 1, а шагов много),

        предотвращает возврат пустого тензора, удерживая последний доступный индекс."""
        sliced_conditioning = []
        for item in conditioning:
            base_tensor = item[0]
            extra_dict = item[1].copy()

            tensor_batch_size = base_tensor.shape[0]

            # Защита от пустого среза (ZeroDivisionError / Пустые картинки)
            actual_start = min(start, tensor_batch_size - 1)
            actual_end = min(end, tensor_batch_size)

            if actual_start >= actual_end:
                actual_start = max(0, tensor_batch_size - 1)
                actual_end = tensor_batch_size

            sliced_tensor = base_tensor[actual_start:actual_end]

            if "pooled_output" in extra_dict:
                pooled = extra_dict["pooled_output"]
                if pooled.shape[0] > 1:
                    extra_dict["pooled_output"] = pooled[actual_start:actual_end]

            sliced_conditioning.append([sliced_tensor, extra_dict])
        return sliced_conditioning

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise, max_micro_batch_size,
               seed_list=None, cfg_list=None, steps_list=None):

        latent_samples = latent_image["samples"]
        total_batch_size = latent_samples.shape[0]
        output_samples = []

        print(f"\n🚀 [MICRO-BATCH] Starting execution matrix loop: Total size {total_batch_size}")

        for start in range(0, total_batch_size, max_micro_batch_size):
            end = min(start + max_micro_batch_size, total_batch_size)
            current_chunk_size = end - start

            # 1. Извлекаем индивидуальный SEED
            if seed_list and start < len(seed_list):
                chunk_seed = int(seed_list[start])
            else:
                chunk_seed = seed + start

            # 2. Извлекаем индивидуальный CFG для этой строки матрицы
            if cfg_list and start < len(cfg_list):
                chunk_cfg = float(cfg_list[start])
            else:
                chunk_cfg = cfg

            # 3. Извлекаем индивидуальный STEPS
            if steps_list and start < len(steps_list):
                chunk_steps = int(steps_list[start])
            else:
                chunk_steps = steps

            # МЕСТО ДЛЯ ПРИНТА: Выводим точные параметры, уходящие в генерацию текущего кадра
            print(f"🎬 [SAMPLER DEBUG] Processing Frame {start + 1}/{total_batch_size} | "
                  f"CFG: {chunk_cfg} | Steps: {chunk_steps} | Seed: {chunk_seed}")

            # 4. Нарезка данных
            chunk_latent = {"samples": latent_samples[start:end]}
            chunk_positive = self.slice_conditioning(positive, start, end)
            chunk_negative = self.slice_conditioning(negative, start, end)

            # 5. Вызов нативного ядра генерации
            chunk_output = nodes.common_ksampler(
                model, chunk_seed, chunk_steps, chunk_cfg, sampler_name, scheduler,
                chunk_positive, chunk_negative, chunk_latent, denoise=denoise
            )

            output_samples.append(chunk_output[0]["samples"])

        merged_samples = torch.cat(output_samples, dim=0)
        # КРИТИЧЕСКИЙ ФИКС: Обязательно возвращаем КОРТЕЖ из одного элемента (словаря)
        return ({"samples": merged_samples},)