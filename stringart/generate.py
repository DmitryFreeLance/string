import os
import argparse
import numpy as np
from math import atan2
from time import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageDraw, ImageFont
from skimage.draw import line_aa, ellipse_perimeter
from skimage.transform import resize
import pandas as pd

# ---------- пути к ресурсам ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resource_path(*parts):
    return os.path.join(SCRIPT_DIR, *parts)

def _load_font(font_size):
    local = _resource_path("images", "arial.ttf")
    if os.path.exists(local):
        try:
            return ImageFont.truetype(local, font_size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()

# ---------- утилиты ----------
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2
    if short_edge == 0:
        return image[:, long_edge_center - short_edge_half : long_edge_center + short_edge_half]
    else:
        return image[long_edge_center - short_edge_half : long_edge_center + short_edge_half, :]

def create_rectangle_nail_positions(shape, nail_step=3):
    h, w = shape
    nails_top = [(0, i) for i in range(0, w, nail_step)]
    nails_bot = [(h - 1, i) for i in range(0, w, nail_step)]
    nails_right = [(i, w - 1) for i in range(0, h, nail_step)][1:-1]
    nails_left  = [(i, 0)     for i in range(0, h, nail_step)][1:-1]
    nails = nails_top + nails_right + nails_bot + nails_left
    print(len(nails), flush=True)
    return np.array(nails)

def create_circle_nail_positions(shape, nail_step=3, r1_multip=1, r2_multip=1):
    h, w = shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 2 - 1
    rr, cc = ellipse_perimeter(cy, cx, int(r * r1_multip), int(r * r2_multip))
    nails = list(set([(rr[i], cc[i]) for i in range(len(cc))]))
    nails.sort(key=lambda c: atan2(c[0] - cy, c[1] - cx))
    nails = nails[::nail_step]
    return np.asarray(nails)

def init_canvas(shape, black=False):
    return (np.zeros(shape) if black else np.ones(shape))

def get_aa_line(p1, p2, strength, pic):
    rr, cc, val = line_aa(p1[0], p1[1], p2[0], p2[1])
    line = pic[rr, cc] + strength * val
    line = np.clip(line, 0, 1)
    return line, rr, cc

def find_best_nail_position(current_position, nails, str_pic, orig_pic, strength):
    best_impr = -1e18
    best_pos = None
    best_idx = None

    if args.random_nails is not None:
        ids = np.random.choice(range(len(nails)), size=args.random_nails, replace=False)
        iterator = list(zip(ids, nails[ids]))
    else:
        iterator = enumerate(nails)

    for idx, pos in iterator:
        over, rr, cc = get_aa_line(current_position, pos, strength, str_pic)
        before = np.abs(str_pic[rr, cc] - orig_pic[rr, cc]) ** 2
        after  = np.abs(over         - orig_pic[rr, cc]) ** 2
        impr = np.sum(before - after)
        if impr >= best_impr:
            best_impr = impr
            best_pos  = pos
            best_idx  = idx

    return best_idx, best_pos, best_impr

def create_art(nails, orig_pic, str_pic, strength, i_limit=None,
               progress_offset=0, progress_total=None):
    start = time()
    iter_times = []
    current_position = nails[0]
    order = [0]
    i = 0
    fails = 0
    last_percent = -1

    while True:
        t0 = time()
        i += 1

        if i_limit is None:
            if fails >= 3: break
        else:
            if i > i_limit: break
            percent = int(((progress_offset + i) / progress_total) * 100) if progress_total else int(i / i_limit * 100)
            if percent > last_percent:
                last_percent = percent
                print(f"{percent}%", flush=True)

        idx, best_pos, best_impr = find_best_nail_position(current_position, nails, str_pic, orig_pic, strength)
        if best_impr <= 0:
            fails += 1
            continue

        order.append(idx)
        over, rr, cc = get_aa_line(current_position, best_pos, strength, str_pic)
        str_pic[rr, cc] = over
        current_position = best_pos
        iter_times.append(time() - t0)

    print(f"Time: {time() - start}", flush=True)
    if iter_times:
        print(f"Avg iteration time: {np.mean(iter_times)}", flush=True)
    return order

def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio * n[0]), int(x_ratio * n[1])) for n in nails]

def pull_order_to_array_bw(order, canvas, nails, strength):
    for s, e in zip(order, order[1:]):
        rr, cc, val = line_aa(nails[s][0], nails[s][1], nails[e][0], nails[e][1])
        canvas[rr, cc] += val * strength
    return np.clip(canvas, 0, 1)

def _to_rgb_255(a):
    return [round(a[0]*255), round(a[1]*255), round(a[2]*255)]

def _to_rgb_binary_string(a):
    r, g, b = _to_rgb_255(a)
    r = 255 if r >= 128 else 0
    g = 255 if g >= 128 else 0
    b = 255 if b >= 128 else 0
    return f"{r},{g},{b}"

def gradToABCD(num):
    cnt = len(nails)
    new = num % (cnt // 4)
    if num < (cnt // 4) + 1: return f'A{new + 1}'
    if num > (cnt // 4) and num < (cnt // 2) + 1: return f'B{new + 1}'
    if num > cnt // 2 and num < (cnt // 4 * 3) + 1: return f'C{new + 1}'
    if num > cnt // 4 * 3: return f'D{new + 1}'

def rectToABCD(num):
    cnt = len(nails)
    new = num % (cnt // 4)
    if num < (cnt // 4) + 1: return f'A{new + 1}'
    if num > (cnt // 4) and num < (cnt // 2) + 1: return f'B{new + 1}'
    if num > cnt // 2 and num < (cnt // 4 * 3) + 1: return f'C{105 - new}'
    if num > cnt // 4 * 3: return f'D{105 - new}'

def pull_order_to_array_rgb(orders, canvas, nails, colors, strength, isRect):
    iters = [iter(zip(o, o[1:])) for o in orders]
    color_order = []

    for _ in range(len(orders[0]) - 1):
        for color_idx, it in enumerate(iters):
            s, e = next(it)
            rr, cc, val = line_aa(nails[s][0], nails[s][1], nails[e][0], nails[e][1])

            canvas[rr, cc] += colors[color_idx] * strength

            pull_colors = [_to_rgb_binary_string(pt) for pt in canvas[rr, cc]]
            main_color = Counter(pull_colors).most_common(1)[0][0] if pull_colors else '-1,-1,-1'

            pull = f"{rectToABCD(s)} -> {rectToABCD(e)}" if isRect else f"{gradToABCD(s)} -> {gradToABCD(e)}"
            color_order.append([pull, main_color])

    print('Кол-во нитей:', len(color_order), flush=True)
    unique_colors = sorted(set([c[1] for c in color_order]))
    print('Кол-во уникальных цветов:', len(unique_colors), flush=True)
    for color in unique_colors:
        try:
            r, g, b = color.split(',')
            print(f"\033[48;2;{r};{g};{b}m{color}\033[0m", flush=True)
        except Exception:
            print(color, flush=True)

    return [np.clip(canvas, 0, 1), color_order]

def add_fields(file_path, isRect, nails):
    blank_path = _resource_path('images', 'blank.jpg')
    if not os.path.exists(blank_path):
        print(f"WARNING: blank.jpg not found at {blank_path}", flush=True)
        return

    blank = Image.open(blank_path).convert('RGBA')
    picture = Image.open(file_path).convert('RGBA')

    field_size = picture.size[0] * 0.12
    blank = blank.resize(
        (int(picture.size[0] + field_size * 2), int(picture.size[1] + field_size * 2)),
        Image.Resampling.NEAREST
    )
    height, width = picture.size

    if not isRect:
        pic_rgb = picture.convert('RGB')
        lum = Image.new('L', [height, width], 0)
        draw = ImageDraw.Draw(lum)
        draw.pieslice([(0, 0), (height, width)], 0, 360, fill=255, outline="white")
        final = np.dstack((np.array(pic_rgb), np.array(lum)))
        picture = Image.fromarray(final).rotate(-0.35)

    position = (int(field_size), int(field_size))
    blank.alpha_composite(picture, position)

    res_h, res_w = blank.size
    c = res_h // 1000 + 1
    drawer = ImageDraw.Draw(blank)
    offset = field_size // 3

    scaled = scale_nails(1.1, 1.1, nails)
    ABCD_font_size = res_h // 24
    p = width // 20

    font_size = res_h // 160
    font = _load_font(font_size)
    fw, fh = int(font_size // 3.1), int(font_size // 1.8)

    if isRect:
        ABCD = {'D': [(offset, res_w // 2), 'red'],
                'C': [(res_h // 2, res_w - offset), 'orange'],
                'B': [(res_h - offset, res_w // 2), 'blue'],
                'A': [(res_h // 2, offset), 'green']}
        n = len(nails)
        for i in range(n):
            xn, yn = nails[i][0] + field_size, nails[i][1] + field_size
            drawer.ellipse((xn - c, yn - c, xn + c, yn + c), fill='red')
            x, y = scaled[i][0] + field_size - p, scaled[i][1] + field_size - p
            if i in range(1, n // 4 + 1):
                drawer.text((x - fw, y - fh), f"{n // 4 - i + 1}", font=font, fill=ABCD['D'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(n // 4 + 1, n // 2):
                if i % 2 == 0:
                    drawer.text((x - fw, y - fh + 12), f"{n // 2 - i + 1}", font=font, fill=ABCD['C'][1]); drawer.line((xn, yn, x, y + 12), fill='gray', width=1)
                else:
                    drawer.text((x - fw, y - fh), f"{n // 2 - i + 1}", font=font, fill=ABCD['C'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i == (n // 4) * 3:
                drawer.text((x - fw, y - fh), f"1", font=font, fill=ABCD['C'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(n // 2, (n // 4) * 3):
                drawer.text((x - fw, y - fh), f"{i - 207}", font=font, fill=ABCD['B'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i == 0:
                drawer.text((x - fw, y - fh), f"{i + 1}", font=font, fill=ABCD['A'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range((n // 4) * 3 + 1, n):
                if i % 2 == 0:
                    drawer.text((x - fw, y - fh), f"{abs((n // 4) * 3 - 1 - i)}", font=font, fill=ABCD['A'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
                else:
                    drawer.text((x - fw, y - fh - 12), f"{abs((n // 4) * 3 - 1 - i)}", font=font, fill=ABCD['A'][1]); drawer.line((xn, yn, x, y - 12), fill='gray', width=1)
    else:
        ABCD = {'A': [(offset, offset), 'green'],
                'B': [(res_h - offset, offset), 'blue'],
                'C': [(res_h - offset, res_w - offset), 'orange'],
                'D': [(offset, res_w - offset), 'red']}
        n = len(nails)
        for i in range(n):
            xn, yn = nails[i][0] + field_size, nails[i][1] + field_size
            drawer.ellipse((xn - c, yn - c, xn + c, yn + c), fill='red')
            x, y = scaled[i][0] + field_size - p, scaled[i][1] + field_size - p
            if i in range(0, n // 4):
                drawer.text((x - fw, y - fh), f"{n // 4 - i}", font=font, fill=ABCD['A'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(n // 4, n // 2):
                drawer.text((x - fw, y - fh), f"{n // 2 - i}", font=font, fill=ABCD['D'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(n // 2, (n // 4) * 3):
                drawer.text((x - fw, y - fh), f"{abs(i - (n // 4) * 3)}", font=font, fill=ABCD['C'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range((n // 4) * 3, n):
                drawer.text((x - fw, y - fh), f"{abs(i - n)}", font=font, fill=ABCD['B'][1]); drawer.line((xn, yn, x, y), fill='gray', width=1)

    big_font = _load_font(ABCD_font_size)
    for letter in ABCD:
        x, y = ABCD[letter][0][0] - ABCD_font_size // 3.1, ABCD[letter][0][1] - ABCD_font_size // 1.8
        drawer.text((x, y), letter, font=big_font, fill=ABCD[letter][1])

    blank.convert('RGB').save(file_path)

# ---------- main ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Create String Art')
    p.add_argument('-i', dest="input_file")
    p.add_argument('-o', dest="output_file", default="output.png")
    p.add_argument('-d', type=int, dest="side_len", default=4000)
    p.add_argument('-s', type=float, dest="export_strength", default=0.1)
    p.add_argument('-l', type=int, dest="pull_amount", default=None)
    p.add_argument('-r', type=int, dest="random_nails", default=None)
    p.add_argument('-r1', type=float, dest="radius1_multiplier", default=1)
    p.add_argument('-r2', type=float, dest="radius2_multiplier", default=1)
    p.add_argument('-n', type=int, dest="nail_step", default=3)
    p.add_argument('-longside', type=int, dest="long_side", default=385)
    p.add_argument('--wb', action="store_true")
    p.add_argument('--rgb', action="store_true")
    p.add_argument('--rect', action="store_true")
    args = p.parse_args()

    img = mpimg.imread(args.input_file)
    if np.any(img > 100):
        img = img / 255.0

    if args.rect:
        LONG_SIDE = 313
        if args.radius1_multiplier == 1 and args.radius2_multiplier == 1:
            img = largest_square(img); img = resize(img, (LONG_SIDE, LONG_SIDE))
        shape = (len(img), len(img[0]))
        nails = create_rectangle_nail_positions(shape, args.nail_step)
    else:
        LONG_SIDE = args.long_side
        if args.radius1_multiplier == 1 and args.radius2_multiplier == 1:
            img = largest_square(img); img = resize(img, (LONG_SIDE, LONG_SIDE))
        shape = (len(img), len(img[0]))
        nails = create_circle_nail_positions(shape, args.nail_step, args.radius1_multiplier, args.radius2_multiplier)

    print(f"Nails amount: {len(nails)}", flush=True)

    if args.rgb:
        iter_strength = -0.1 if args.wb else 0.1

        r = img[:, :, 0]; g = img[:, :, 1]; b = img[:, :, 2]
        total_for_progress = (args.pull_amount or 0) * 3 if args.pull_amount else None

        str_r = init_canvas(shape, black=not args.wb)
        order_r = create_art(nails, r, str_r, iter_strength, i_limit=args.pull_amount,
                             progress_offset=0, progress_total=total_for_progress)

        str_g = init_canvas(shape, black=not args.wb)
        order_g = create_art(nails, g, str_g, iter_strength, i_limit=args.pull_amount,
                             progress_offset=(args.pull_amount or 0), progress_total=total_for_progress)

        str_b = init_canvas(shape, black=not args.wb)
        order_b = create_art(nails, b, str_b, iter_strength, i_limit=args.pull_amount,
                             progress_offset=(args.pull_amount or 0) * 2, progress_total=total_for_progress)

        max_pulls = np.max([len(order_r), len(order_g), len(order_b)])
        order_r = order_r + [order_r[-1]] * (max_pulls - len(order_r))
        order_g = order_g + [order_g[-1]] * (max_pulls - len(order_g))
        order_b = order_b + [order_b[-1]] * (max_pulls - len(order_b))
        orders = [order_r, order_g, order_b]

        color_dims = (int(args.side_len * args.radius1_multiplier),
                      int(args.side_len * args.radius2_multiplier), 3)
        blank = init_canvas(color_dims, black=not args.wb)
        nails_img = scale_nails(color_dims[1] / shape[1], color_dims[0] / shape[0], nails)

        result, instruction = pull_order_to_array_rgb(
            orders, blank, nails_img,
            (np.array((1., 0., 0.)), np.array((0., 1., 0.)), np.array((0., 0., 1.))),
            -args.export_strength if args.wb else args.export_strength,
            args.rect
        )

        # изображение
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        plt.imsave(args.output_file, np.clip(result, 0.0, 1.0))

        # таблица
        colors_map = {
            '255,0,255': 'Фиолетовый',
            '0,255,255': 'Голубой',
            '255,255,0': 'Жёлтый',
            '0,0,255': 'Синий',
            '0,255,0': 'Зелёный',
            '255,0,0': 'Красный',
            '0,0,0': 'Чёрный',
            '255,255,255': 'Белый',
            '-1,-1,-1': 'Нет цвета'
        }
        df = pd.DataFrame([[p[0], colors_map.get(p[1], p[1]), p[1]] for p in instruction],
                          columns=['Нить', 'Цвет', 'RGB'])

        xlsx_path = f"{args.output_file}_instruction.xlsx"
        saved_path = None
        # openpyxl
        try:
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
            saved_path = xlsx_path
        except Exception:
            # xlsxwriter
            try:
                with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False)
                saved_path = xlsx_path
            except Exception:
                # csv запасной
                csv_path = f"{args.output_file}_instruction.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                saved_path = csv_path

        # поля и метки (если есть blank.jpg)
        add_fields(args.output_file, args.rect, nails_img)

        # сообщаем боту фактический путь
        if saved_path:
            print(f"INSTR: {os.path.abspath(saved_path)}", flush=True)
            # для обратной совместимости
            print(f"XLSX: {os.path.abspath(saved_path)}", flush=True)
        print("100%", flush=True)

    else:
        orig = rgb2gray(img) * 0.9
        dims = (int(args.side_len * args.radius1_multiplier),
                int(args.side_len * args.radius2_multiplier))

        if args.wb:
            str_pic = init_canvas(shape, black=False)
            order = create_art(nails, orig, str_pic, -0.05, i_limit=args.pull_amount,
                               progress_offset=0, progress_total=args.pull_amount)
            blank = init_canvas(dims, black=False)
            strength = -args.export_strength
        else:
            str_pic = init_canvas(shape, black=True)
            order = create_art(nails, orig, str_pic, 0.05, i_limit=args.pull_amount,
                               progress_offset=0, progress_total=args.pull_amount)
            blank = init_canvas(dims, black=True)
            strength = args.export_strength

        nails_img = scale_nails(dims[1] / shape[1], dims[0] / shape[0], nails)
        result = pull_order_to_array_bw(order, blank, nails_img, strength)

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        mpimg.imsave(args.output_file, result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)
        print("100%", flush=True)