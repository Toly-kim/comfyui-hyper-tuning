import os
import re
from PIL import Image, ImageDraw
from src.utils.paths import paths


def create_grid_for_pair(path1, path2, pair_idx_plus_seed, filename1, filename2):
    """
    Создает склейку двух изображений с подписями.
    """
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    # Создаем полотно (двойная ширина + место под заголовок 40px)
    grid = Image.new('RGB', (img1.width + img2.width, img1.height + 40), (255, 255, 255))
    grid.paste(img1, (0, 40))
    grid.paste(img2, (img1.width, 40))

    # Рисуем подписи
    draw = ImageDraw.Draw(grid)
    draw.text((10, 10), f"LEFT: {os.path.basename(path1)}", fill=(0, 0, 0))
    draw.text((img1.width + 10, 10), f"RIGHT: {os.path.basename(path2)}", fill=(0, 0, 0))

    # Формируем имя и сохраняем
    # out_name = f"pair_{pair_idx_plus_seed:02d}_{filename1}_vs_{filename2}.png"
    out_name = f"pair_{pair_idx_plus_seed}_{filename1}_vs_{filename2}.png"

    # config_path = paths.get_config_path("config.yaml")
    # cfg = utils.paths.load_yaml_config(config_path)
    cfg = paths.get_cfg()
    paths_config = cfg['paths']
    output_dir = paths_config['output_grid']

    full_path = os.path.join(output_dir, out_name)
    grid.save(full_path)

    return full_path

def get_file_parts(filename):
    match = re.match(r"(.*?)(\d+)(.*)", filename)
    if not match:
        raise ValueError(f"Не удалось распознать формат имени: {filename}")
    prefix, number, suffix = match.groups()
    return prefix, int(number), suffix

def create_sequential_grids():
    # config_path = paths.get_config_path("config.yaml")
    # cfg = utils.paths.load_yaml_config(config_path)
    cfg = paths.get_cfg()
    paths_config = cfg['paths']
    number_of_runs = cfg['runs']['count']

    prefix1, start_num1, suffix1 = get_file_parts(paths_config["start_left_img"])
    prefix2, start_num2, suffix2 = get_file_parts(paths_config["start_right_img"])

    for i in range(number_of_runs):
        # Генерируем имена текущей пары
        # :05d означает: целое число, минимум 5 знаков, заполнить нулями
        name1 = f"{prefix1}{start_num1 + i:05d}{suffix1}"
        name2 = f"{prefix2}{start_num2 + i:05d}{suffix2}"

        path1 = os.path.join(paths_config['input_drive'], name1)
        path2 = os.path.join(paths_config['input_drive'], name2)

        if os.path.exists(path1) and os.path.exists(path2):
            try:
                grid_res = create_grid_for_pair(path1, path2, i + 1, start_num1 + i, start_num2 + i)
                print(f"Pair {i+1} is done in {os.path.basename(grid_res)}")
            except Exception as e:
                print(f"Error when making a pair {i}: {e}")
        else:
            print(f"⚠️ Files not found ({name1} or {name2})")

if __name__ == "__main__":
    create_sequential_grids() # Make 2 img in 1 grid