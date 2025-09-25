import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import line_aa, ellipse_perimeter
from math import atan2
from skimage.transform import resize
from time import time
import argparse
from collections import Counter
import pandas as pd


# =============================== УТИЛИТЫ ===============================

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])  # 0 = vertical <= horizontal; 1 = otherwise
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2
    if short_edge == 0:
        return image[:, long_edge_center - short_edge_half:
                        long_edge_center + short_edge_half]
    if short_edge == 1:
        return image[long_edge_center - short_edge_half:
                     long_edge_center + short_edge_half, :]


def create_rectangle_nail_positions(shape, nail_step=3):
    height, width = shape

    nails_top = [(0, i) for i in range(0, width, nail_step)]
    nails_bot = [(height - 1, i) for i in range(0, width, nail_step)]
    nails_right = [(i, width - 1) for i in range(0, height, nail_step)][1:-1]
    nails_left = [(i, 0) for i in range(0, height, nail_step)][1:-1]

    nails = nails_top + nails_right + nails_bot + nails_left
    print(len(nails))
    return np.array(nails)


def create_circle_nail_positions(shape, nail_step=3, r1_multip=1, r2_multip=1):
    height = shape[0]
    width = shape[1]

    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1
    rr, cc = ellipse_perimeter(centre[0], centre[1], int(radius * r1_multip), int(radius * r2_multip))
    nails = list(set([(rr[i], cc[i]) for i in range(len(cc))]))
    nails.sort(key=lambda c: atan2(c[0] - centre[0], c[1] - centre[1]))
    nails = nails[::nail_step]

    return np.asarray(nails)


def init_canvas(shape, black=False):
    # white background => black=False; black background => black=True
    return np.zeros(shape) if black else np.ones(shape)


def get_aa_line(from_pos, to_pos, str_strength, picture):
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)
    return line, rr, cc


def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength, random_nails=None):
    best_cumulative_improvement = -1e18
    best_nail_position = None
    best_nail_idx = None

    if random_nails is not None:
        nail_ids = np.random.choice(range(len(nails)), size=min(random_nails, len(nails)), replace=False)
        nails_and_ids = list(zip(nail_ids, nails[nail_ids]))
    else:
        nails_and_ids = enumerate(nails)

    for nail_idx, nail_position in nails_and_ids:
        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)

        before_overlayed_line_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc]) ** 2
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr, cc]) ** 2

        cumulative_improvement = np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement


def create_art(nails, orig_pic, str_pic, str_strength, i_limit=None,
               progress_offset=0, progress_total=None, random_nails=None):
    start = time()
    iter_times = []

    current_position = nails[0]
    pull_order = [0]

    i = 0
    fails = 0
    last_percent = -1

    while True:
        start_iter = time()
        i += 1

        if i_limit is None:
            if i % 500 == 0:
                print(f"Iteration {i}", flush=True)
        else:
            if i > i_limit:
                break
            if progress_total:
                percent = int(((progress_offset + i) / progress_total) * 100)
            else:
                percent = int(i / i_limit * 100)
            if percent > last_percent:
                last_percent = percent
                print(f"{percent}%", flush=True)

        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(
            current_position, nails, str_pic, orig_pic, str_strength, random_nails=random_nails
        )

        if i_limit is None:
            # старый «несчётный» режим — додавливаем 3 провала подряд
            if best_cumulative_improvement <= 0:
                fails += 1
                if fails >= 3:
                    break
                continue
        else:
            # при фиксированном лимите — шаг делаем всегда, даже если прироста нет
            pass

        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line

        current_position = best_nail_position
        iter_times.append(time() - start_iter)

        if i_limit is None and i > 20000:
            break

    print(f"Time: {time() - start}", flush=True)
    if iter_times:
        print(f"Avg iteration time: {np.mean(iter_times)}", flush=True)
    return pull_order


def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio * nail[0]), int(x_ratio * nail[1])) for nail in nails]


def pull_order_to_array_bw(order, canvas, nails, strength):
    for pull_start, pull_end in zip(order, order[1:]):  # pairwise iteration
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength
    return np.clip(canvas, a_min=0, a_max=1)


def _snap_255(x):
    """Квантование в {0,255} по порогу 127.5"""
    return 255 if x >= 127.5 else 0


def npToRGB255(np_array):
    """Перевод [0..1] -> {0,255} с квантованием по каждому каналу"""
    r = _snap_255(round(np_array[0] * 255))
    g = _snap_255(round(np_array[1] * 255))
    b = _snap_255(round(np_array[2] * 255))
    return [r, g, b]


def gradToABCD(number, nails_global):
    nails_count = len(nails_global)
    new_number = number % (nails_count // 4)
    if number < (nails_count // 4) + 1:
        return f'A{new_number + 1}'
    if number > (nails_count // 4) and number < (nails_count // 2) + 1:
        return f'B{new_number + 1}'
    if number > nails_count // 2 and number < (nails_count // 4 * 3) + 1:
        return f'C{new_number + 1}'
    if number > nails_count // 4 * 3:
        return f'D{new_number + 1}'


def rectToABCD(number, nails_global):
    nails_count = len(nails_global)
    new_number = number % (nails_count // 4)
    if number < (nails_count // 4) + 1:
        return f'A{new_number + 1}'
    if number > (nails_count // 4) and number < (nails_count // 2) + 1:
        return f'B{new_number + 1}'
    if number > nails_count // 2 and number < (nails_count // 4 * 3) + 1:
        return f'C{105 - new_number}'
    if number > nails_count // 4 * 3:
        return f'D{105 - new_number}'


def pull_order_to_array_rgb(orders, canvas, nails, colors, strength, isRect):
    """
    Рисуем по кругу: R -> G -> B -> ...
    Цвет одной нити в Excel — это МАЖОРИТАРНЫЙ цвет пикселей линии ПОСЛЕ отрисовки,
    но квантованный в {0,255} по каждому каналу, чтобы в таблице были только допустимые цифры.
    """
    color_order_iterators = [iter(zip(order, order[1:])) for order in orders]
    color_order = []

    for _ in range(len(orders[0]) - 1):
        for color_idx, iterator in enumerate(color_order_iterators):
            pull_start, pull_end = next(iterator)
            rr_aa, cc_aa, _ = line_aa(
                nails[pull_start][0], nails[pull_start][1],
                nails[pull_end][0], nails[pull_end][1]
            )

            # рисуем линию цветом канала
            canvas[rr_aa, cc_aa] += colors[color_idx] * strength
            canvas[rr_aa, cc_aa] = np.clip(canvas[rr_aa, cc_aa], 0.0, 1.0)

            # основной цвет линии: берём реальные значения канвы на пикселях линии и квантем в {0,255}
            pull_colors = []
            for point in canvas[rr_aa, cc_aa][:]:
                r, g, b = npToRGB255(point)
                color_string = f'{r},{g},{b}'
                pull_colors.append(color_string)

            pull_main_color = Counter(pull_colors).most_common(5)
            main_color = '-1,-1,-1'
            for cand, _cnt in pull_main_color:
                if '-' not in cand:
                    main_color = cand
                    break

            if isRect:
                pull = f"{rectToABCD(pull_start, nails)} -> {rectToABCD(pull_end, nails)}"
            else:
                pull = f"{gradToABCD(pull_start, nails)} -> {gradToABCD(pull_end, nails)}"

            color_order.append([pull, main_color])

    print('Кол-во нитей:', len(color_order))
    unique_colors = set([color[1] for color in color_order])
    print('Кол-во уникальных цветов:', len(unique_colors))
    for color in unique_colors:
        r, g, b = color.split(',')
        print(f"\033[48;2;{r};{g};{b}m{r},{g},{b}\033[0m")

    return [np.clip(canvas, a_min=0, a_max=1), color_order]


# ====================== Надёжная разметка бланка ======================

class FieldAnnotator:
    """
    Поиск ресурсов относительным путём (images/blank.jpg, arial.ttf).
    Если blank.jpg не найден — разметка пропускается, чтобы не ломать бота.
    """

    def __init__(self, base_file: str, images_subdir: str = "images"):
        self.script_dir = os.path.dirname(os.path.abspath(base_file))
        self.images_dir = os.path.join(self.script_dir, images_subdir)

    def _res_path(self, *parts: str) -> str:
        return os.path.join(self.script_dir, *parts)

    def _img_path(self, *parts: str) -> str:
        return os.path.join(self.images_dir, *parts)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        candidates = [
            self._res_path("arial.ttf"),
            self._img_path("arial.ttf"),
            "arial.ttf",
        ]
        for p in candidates:
            try:
                if os.path.exists(p):
                    return ImageFont.truetype(p, size)
            except Exception:
                pass
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _scale_nails(x_ratio, y_ratio, nails):
        return [(int(y_ratio * n[0]), int(x_ratio * n[1])) for n in nails]

    def add_fields(self, file_path: str, is_rect: bool, nails):
        # ищем blank.jpg
        candidates = [
            self._img_path("blank.jpg"),
            self._res_path("blank.jpg"),
            os.path.join(os.getcwd(), "images", "blank.jpg"),
            os.path.join(os.getcwd(), "blank.jpg"),
        ]
        blank_path = next((p for p in candidates if os.path.exists(p)), None)
        if not blank_path:
            print("WARN: BLANK_NOT_FOUND -> skip add_fields()", flush=True)
            return

        try:
            blank = Image.open(blank_path).convert("RGBA")
        except Exception as e:
            print(f"WARN: cannot open blank: {e} -> skip add_fields()", flush=True)
            return

        try:
            picture = Image.open(file_path).convert("RGBA")
        except Exception as e:
            print(f"WARN: cannot open picture: {e} -> skip add_fields()", flush=True)
            return

        field_size = picture.size[0] * 0.12
        blank = blank.resize(
            (int(picture.size[0] + field_size * 2), int(picture.size[1] + field_size * 2)),
            Image.Resampling.NEAREST,
        )
        height, width = picture.size

        # круглый кроп для круглого поля
        if not is_rect:
            pic_rgb = picture.convert("RGB")
            lum = Image.new("L", [height, width], 0)
            draw = ImageDraw.Draw(lum)
            draw.pieslice([(0, 0), (height, width)], 0, 360, fill=255, outline="white")
            final = np.dstack((np.array(pic_rgb), np.array(lum)))
            picture = Image.fromarray(final).rotate(-0.35)

        position = (int(field_size), int(field_size))
        blank.alpha_composite(picture, position)

        # разметка
        res_h, res_w = blank.size
        c = res_h // 1000 + 1
        drawer = ImageDraw.Draw(blank)
        offset = field_size // 3

        scaled_for_labels = self._scale_nails(1.1, 1.1, nails)
        ABCD_font_size = res_h // 24
        p = width // 20

        font_size = res_h // 160
        font = self._load_font(font_size)
        fw, fh = int(font_size // 3.1), int(font_size // 1.8)

        if is_rect:
            ABCD = {
                "D": [(offset, res_w // 2), "red"],
                "C": [(res_h // 2, res_w - offset), "orange"],
                "B": [(res_h - offset, res_w // 2), "blue"],
                "A": [(res_h // 2, offset), "green"],
            }
            n = len(nails)
            for i in range(n):
                xn, yn = nails[i][0] + field_size, nails[i][1] + field_size
                drawer.ellipse((xn - c, yn - c, xn + c, yn + c), fill="red")
                x, y = scaled_for_labels[i][0] + field_size - p, scaled_for_labels[i][1] + field_size - p
                if i in range(1, n // 4 + 1):
                    drawer.text((x - fw, y - fh), f"{n // 4 - i + 1}", font=font, fill=ABCD["D"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i in range(n // 4 + 1, n // 2):
                    if i % 2 == 0:
                        drawer.text((x - fw, y - fh + 12), f"{n // 2 - i + 1}", font=font, fill=ABCD["C"][1])
                        drawer.line((xn, yn, x, y + 12), fill="gray", width=1)
                    else:
                        drawer.text((x - fw, y - fh), f"{n // 2 - i + 1}", font=font, fill=ABCD["C"][1])
                        drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i == (n // 4) * 3:
                    drawer.text((x - fw, y - fh), "1", font=font, fill=ABCD["C"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i in range(n // 2, (n // 4) * 3):
                    drawer.text((x - fw, y - fh), f"{i - 207}", font=font, fill=ABCD["B"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i == 0:
                    drawer.text((x - fw, y - fh), f"{i + 1}", font=font, fill=ABCD["A"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i in range((n // 4) * 3 + 1, n):
                    if i % 2 == 0:
                        drawer.text((x - fw, y - fh), f"{abs((n // 4) * 3 - 1 - i)}", font=font, fill=ABCD["A"][1])
                        drawer.line((xn, yn, x, y), fill="gray", width=1)
                    else:
                        drawer.text((x - fw, y - fh - 12), f"{abs((n // 4) * 3 - 1 - i)}", font=font, fill=ABCD["A"][1])
                        drawer.line((xn, yn, x, y - 12), fill="gray", width=1)
        else:
            ABCD = {
                "A": [(offset, offset), "green"],
                "B": [(res_h - offset, offset), "blue"],
                "C": [(res_h - offset, res_w - offset), "orange"],
                "D": [(offset, res_w - offset), "red"],
            }
            n = len(nails)
            for i in range(n):
                xn, yn = nails[i][0] + field_size, nails[i][1] + field_size
                drawer.ellipse((xn - c, yn - c, xn + c, yn + c), fill="red")
                x, y = scaled_for_labels[i][0] + field_size - p, scaled_for_labels[i][1] + field_size - p
                if i in range(0, n // 4):
                    drawer.text((x - fw, y - fh), f"{n // 4 - i}", font=font, fill=ABCD["A"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i in range(n // 4, n // 2):
                    drawer.text((x - fw, y - fh), f"{n // 2 - i}", font=font, fill=ABCD["D"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i in range(n // 2, (n // 4) * 3):
                    drawer.text((x - fw, y - fh), f"{abs(i - (n // 4) * 3)}", font=font, fill=ABCD["C"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)
                if i in range((n // 4) * 3, n):
                    drawer.text((x - fw, y - fh), f"{abs(i - n)}", font=font, fill=ABCD["B"][1])
                    drawer.line((xn, yn, x, y), fill="gray", width=1)

        for letter in ABCD:
            big = self._load_font(ABCD_font_size)
            x, y = ABCD[letter][0][0] - ABCD_font_size // 3.1, ABCD[letter][0][1] - ABCD_font_size // 1.8
            drawer.text((x, y), letter, font=big, fill=ABCD[letter][1])

        blank.convert("RGB").save(file_path)


# =============================== MAIN ===============================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create String Art')
    parser.add_argument('-i', action="store", dest="input_file")
    parser.add_argument('-o', action="store", dest="output_file", default="output.png")
    parser.add_argument('-d', action="store", type=int, dest="side_len", default=4000)
    parser.add_argument('-s', action="store", type=float, dest="export_strength", default=0.1)
    parser.add_argument('-l', action="store", type=int, dest="pull_amount", default=None)
    parser.add_argument('-r', action="store", type=int, dest="random_nails", default=None)
    parser.add_argument('-r1', action="store", type=float, dest="radius1_multiplier", default=1)
    parser.add_argument('-r2', action="store", type=float, dest="radius2_multiplier", default=1)
    parser.add_argument('-n', action="store", type=int, dest="nail_step", default=3)
    parser.add_argument('-longside', action="store", type=int, dest="long_side", default=385)
    parser.add_argument('--wb', action="store_true")
    parser.add_argument('--rgb', action="store_true")
    parser.add_argument('--rect', action="store_true")

    args = parser.parse_args()

    img = mpimg.imread(args.input_file)
    if np.any(img > 100):
        img = img / 255.0

    if args.rect:
        LONG_SIDE = 313
        if args.radius1_multiplier == 1 and args.radius2_multiplier == 1:
            img = largest_square(img)
            img = resize(img, (LONG_SIDE, LONG_SIDE))
        shape = (len(img), len(img[0]))
        nails = create_rectangle_nail_positions(shape, args.nail_step)
    else:
        LONG_SIDE = args.long_side
        if args.radius1_multiplier == 1 and args.radius2_multiplier == 1:
            img = largest_square(img)
            img = resize(img, (LONG_SIDE, LONG_SIDE))
        shape = (len(img), len(img[0]))
        nails = create_circle_nail_positions(shape, args.nail_step, args.radius1_multiplier, args.radius2_multiplier)

    print(f"Nails amount: {len(nails)}", flush=True)

    annot = FieldAnnotator(base_file=__file__, images_subdir="images")

    if args.rgb:
        # Белый фон -> линии затемняют -> отрицательная сила
        iteration_strength = -0.1 if args.wb else 0.1

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        total_for_progress = (args.pull_amount or 0) * 3 if args.pull_amount else None

        str_pic_r = init_canvas(shape, black=not args.wb)
        pull_orders_r = create_art(
            nails, r, str_pic_r, iteration_strength,
            i_limit=args.pull_amount,
            progress_offset=0,
            progress_total=total_for_progress,
            random_nails=args.random_nails
        )

        str_pic_g = init_canvas(shape, black=not args.wb)
        pull_orders_g = create_art(
            nails, g, str_pic_g, iteration_strength,
            i_limit=args.pull_amount,
            progress_offset=(args.pull_amount or 0),
            progress_total=total_for_progress,
            random_nails=args.random_nails
        )

        str_pic_b = init_canvas(shape, black=not args.wb)
        pull_orders_b = create_art(
            nails, b, str_pic_b, iteration_strength,
            i_limit=args.pull_amount,
            progress_offset=(args.pull_amount or 0) * 2,
            progress_total=total_for_progress,
            random_nails=args.random_nails
        )

        # выравниваем длины (как в стоке)
        max_pulls = np.max([len(pull_orders_r), len(pull_orders_g), len(pull_orders_b)])
        pull_orders_r = pull_orders_r + [pull_orders_r[-1]] * (max_pulls - len(pull_orders_r))
        pull_orders_g = pull_orders_g + [pull_orders_g[-1]] * (max_pulls - len(pull_orders_g))
        pull_orders_b = pull_orders_b + [pull_orders_b[-1]] * (max_pulls - len(pull_orders_b))

        pull_orders = [pull_orders_r, pull_orders_g, pull_orders_b]

        color_image_dimens = int(args.side_len * args.radius1_multiplier), int(args.side_len * args.radius2_multiplier), 3
        blank = init_canvas(color_image_dimens, black=not args.wb)

        scaled_nails = scale_nails(
            color_image_dimens[1] / shape[1],
            color_image_dimens[0] / shape[0],
            nails
        )

        result, instruction = pull_order_to_array_rgb(
            pull_orders,
            blank,
            scaled_nails,
            (np.array((1., 0., 0.,)), np.array((0., 1., 0.,)), np.array((0., 0., 1.,))),
            -args.export_strength if args.wb else args.export_strength,
            args.rect
        )

        # только допустимые комбинации {0,255}^3
        colors_map = {
            '255,0,0': 'Красный',
            '0,255,0': 'Зеленый',
            '0,0,255': 'Синий',
            '255,255,0': 'Желтый',
            '0,255,255': 'Голубой',
            '255,0,255': 'Фиолетовый',
            '0,0,0': 'Черный',
            '255,255,255': 'Белый',
            '-1,-1,-1': 'Нет цвета',
        }

        # сохраняем PNG
        plt.imsave(args.output_file, np.clip(result, 0.0, 1.0))

        # сохраняем XLSX: Нить / Цвет / RGB
        df = pd.DataFrame(
            [[pair[0], colors_map.get(pair[1], pair[1]), pair[1]] for pair in instruction],
            columns=['Нить', 'Цвет', 'RGB']
        )
        df.to_excel(f"{args.output_file}_instruction.xlsx", index=False)

        # разметка бланка (не критична — если нет blank.jpg, просто пропустим)
        annot.add_fields(file_path=args.output_file, is_rect=args.rect, nails=scaled_nails)

        print("100%", flush=True)

    else:
        # Grayscale
        orig_pic = rgb2gray(img) * 0.9

        image_dimens = int(args.side_len * args.radius1_multiplier), int(args.side_len * args.radius2_multiplier)

        if args.wb:
            # белая канва, линии затемняют
            str_pic = init_canvas(shape, black=False)
            pull_order = create_art(
                nails, orig_pic, str_pic, -0.05, i_limit=args.pull_amount,
                progress_offset=0, progress_total=args.pull_amount, random_nails=args.random_nails
            )
            blank = init_canvas(image_dimens, black=False)
            strength = -args.export_strength
        else:
            # чёрная канва, линии светлеют
            str_pic = init_canvas(shape, black=True)
            pull_order = create_art(
                nails, orig_pic, str_pic, 0.05, i_limit=args.pull_amount,
                progress_offset=0, progress_total=args.pull_amount, random_nails=args.random_nails
            )
            blank = init_canvas(image_dimens, black=True)
            strength = args.export_strength

        scaled_nails = scale_nails(
            image_dimens[1] / shape[1],
            image_dimens[0] / shape[0],
            nails
        )

        result = pull_order_to_array_bw(pull_order, blank, scaled_nails, strength)
        mpimg.imsave(args.output_file, result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)

        # разметка бланка
        annot.add_fields(file_path=args.output_file, is_rect=args.rect, nails=scaled_nails)

        print("100%", flush=True)