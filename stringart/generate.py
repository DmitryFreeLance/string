import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import line_aa, ellipse_perimeter
from math import *
from skimage.transform import resize
from time import time
import argparse
from collections import Counter
import pandas as pd


# from convert_image import convert

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
    # ВАЖНО: white background => black=False
    if black:
        return np.zeros(shape)
    else:
        return np.ones(shape)


def get_aa_line(from_pos, to_pos, str_strength, picture):
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)
    return line, rr, cc


def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength):
    best_cumulative_improvement = -99999
    best_nail_position = None
    best_nail_idx = None

    if args.random_nails != None:
        nail_ids = np.random.choice(range(len(nails)), size=args.random_nails, replace=False)
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
               progress_offset=0, progress_total=None):
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

        if i % 500 == 0:
            print(f"Iteration {i}")

        if i_limit == None:
            if fails >= 3:
                break
        else:
            if i > i_limit:
                break
            # прогресс для бота (если progress_total задан — сквозной по каналам)
            percent = int(((progress_offset + i) / progress_total) * 100) if progress_total else int(i / i_limit * 100)
            if percent > last_percent:
                last_percent = percent
                print(f"{percent}%", flush=True)

        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(
            current_position, nails, str_pic, orig_pic, str_strength
        )

        if best_cumulative_improvement <= 0:
            fails += 1
            continue

        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line

        current_position = best_nail_position
        iter_times.append(time() - start_iter)

    print(f"Time: {time() - start}")
    print(f"Avg iteration time: {np.mean(iter_times)}")
    return pull_order


def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio * nail[0]), int(x_ratio * nail[1])) for nail in nails]


def pull_order_to_array_bw(order, canvas, nails, strength):
    for pull_start, pull_end in zip(order, order[1:]):  # pairwise iteration
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength
    return np.clip(canvas, a_min=0, a_max=1)


def npToRGB(np_array):
    return [round(np_array[0] * 255), round(np_array[1] * 255), round(np_array[2] * 255)]


def gradToABCD(number):
    nails_count = len(nails)
    new_number = number % (nails_count // 4)
    if number < (nails_count // 4) + 1:
        return f'A{new_number + 1}'
    if number > (nails_count // 4) and number < (nails_count // 2) + 1:
        return f'B{new_number + 1}'
    if number > nails_count // 2 and number < (nails_count // 4 * 3) + 1:
        return f'C{new_number + 1}'
    if number > nails_count // 4 * 3:
        return f'D{new_number + 1}'


def rectToABCD(number):
    nails_count = len(nails)
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
    color_order_iterators = [iter(zip(order, order[1:])) for order in orders]
    color_order = []
    pull_number = 0

    for _ in range(len(orders[0]) - 1):
        # pull colors alternately
        for color_idx, iterator in enumerate(color_order_iterators):
            pull_start, pull_end = next(iterator)
            rr_aa, cc_aa, val_aa = line_aa(
                nails[pull_start][0], nails[pull_start][1],
                nails[pull_end][0], nails[pull_end][1]
            )
            pull_number += 1

            # рисуем линию цветом канала
            canvas[rr_aa, cc_aa] += colors[color_idx] * strength

            # вычисляем основной цвет линии по фактическим RGB на пикселях линии
            pull_colors = []
            for point in canvas[rr_aa, cc_aa][:]:
                r, g, b = npToRGB(point)
                color_string = f'{r},{g},{b}'
                pull_colors.append(color_string)
            pull_main_color = Counter(pull_colors).most_common(5)
            for c in range(len(pull_main_color)):
                main_color = pull_main_color[c][0]
                if '-' not in main_color:
                    break

            if isRect:
                pull = f"{rectToABCD(pull_start)} -> {rectToABCD(pull_end)}"
            else:
                pull = f"{gradToABCD(pull_start)} -> {gradToABCD(pull_end)}"
            if '-' in main_color:
                main_color = '-1,-1,-1'
            color_order.append([pull, main_color])

    print('Кол-во нитей:', len(color_order))
    unique_colors = set([color[1] for color in color_order])
    print('Кол-во уникальных цветов:', len(unique_colors))
    for color in unique_colors:
        r, g, b = color.split(',')
        print(f"\033[48;2;{r};{g};{b}m{r},{g},{b}\033[0m")

    return [np.clip(canvas, a_min=0, a_max=1), color_order]


def add_fields(file_path, isRect, nails):
    blank = Image.open('images/blank.jpg')
    picture = Image.open(file_path)

    blank = blank.convert('RGBA')
    picture = picture.convert('RGBA')
    field_size = picture.size[0] * 0.12
    blank = blank.resize((int(picture.size[0] + field_size * 2), int(picture.size[1] + field_size * 2)),
                         Image.Resampling.NEAREST)
    height, width = picture.size

    # circle crop
    if not isRect:
        picture = picture.convert('RGB')
        lum_img = Image.new('L', [height, width], 0)

        draw = ImageDraw.Draw(lum_img)
        draw.pieslice([(0, 0), (height, width)], 0, 360,
                      fill=255, outline="white")
        img_arr = np.array(picture)
        lum_img_arr = np.array(lum_img)
        final_img_arr = np.dstack((img_arr, lum_img_arr))
        picture = Image.fromarray(final_img_arr)
        picture = picture.rotate(-0.35)

    position = (int(field_size), int(field_size))
    blank.alpha_composite(picture, position)

    # разметка бланка
    res_height, res_width = blank.size
    c = res_height // 1000 + 1  # correction
    drawer = ImageDraw.Draw(blank)
    offset = field_size // 3

    scaled_nails = scale_nails(1.1, 1.1, nails)
    ABCD_font_size = res_height // 24
    p = width // 20

    font_size = res_height // 160  # 136
    font = ImageFont.truetype("arial.ttf", font_size)
    font_cor_w = font_size // 3.1
    font_cor_h = font_size // 1.8

    if isRect:
        ABCD = {
            'D': [(offset, res_width // 2), 'red'],
            'C': [(res_height // 2, res_width - offset), 'orange'],
            'B': [(res_height - offset, res_width // 2), 'blue'],
            'A': [(res_height // 2, offset), 'green'],
        }
        nails_count = len(nails)
        for i in range(nails_count):
            xn, yn = nails[i][0] + field_size, nails[i][1] + field_size
            drawer.ellipse((xn - c, yn - c, xn + c, yn + c), fill='red')  # гвозди
            x, y = scaled_nails[i][0] + field_size - p, scaled_nails[i][1] + field_size - p
            if i in range(1, nails_count // 4 + 1):
                drawer.text((x - font_cor_w, y - font_cor_h), f"{nails_count // 4 - i + 1}", font=font,
                            fill=ABCD['D'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(nails_count // 4 + 1, nails_count // 2):
                if i % 2 == 0:
                    drawer.text((x - font_cor_w, y - font_cor_h + 12), f"{nails_count // 2 - i + 1}", font=font,
                                fill=ABCD['C'][1])
                    drawer.line((xn, yn, x, y + 12), fill='gray', width=1)
                else:
                    drawer.text((x - font_cor_w, y - font_cor_h), f"{nails_count // 2 - i + 1}", font=font,
                                fill=ABCD['C'][1])
                    drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i == (nails_count // 4) * 3:
                drawer.text((x - font_cor_w, y - font_cor_h), f"{1}", font=font, fill=ABCD['C'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(nails_count // 2, (nails_count // 4) * 3):
                drawer.text((x - font_cor_w, y - font_cor_h), f"{i - 207}", font=font, fill=ABCD['B'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i == 0:
                drawer.text((x - font_cor_w, y - font_cor_h), f"{i + 1}", font=font, fill=ABCD['A'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range((nails_count // 4) * 3 + 1, nails_count):
                if i % 2 == 0:
                    drawer.text((x - font_cor_w, y - font_cor_h), f"{abs((nails_count // 4) * 3 - 1 - i)}", font=font,
                                fill=ABCD['A'][1])
                    drawer.line((xn, yn, x, y), fill='gray', width=1)
                else:
                    drawer.text((x - font_cor_w, y - font_cor_h - 12), f"{abs((nails_count // 4) * 3 - 1 - i)}",
                                font=font, fill=ABCD['A'][1])
                    drawer.line((xn, yn, x, y - 12), fill='gray', width=1)
    else:
        ABCD = {
            'A': [(offset, offset), 'green'],
            'B': [(res_height - offset, offset), 'blue'],
            'C': [(res_height - offset, res_width - offset), 'orange'],
            'D': [(offset, res_width - offset), 'red'],
        }
        nails_count = len(nails)
        for i in range(nails_count):
            xn, yn = nails[i][0] + field_size, nails[i][1] + field_size
            drawer.ellipse((xn - c, yn - c, xn + c, yn + c), fill='red')  # гвозди
            x, y = scaled_nails[i][0] + field_size - p, scaled_nails[i][1] + field_size - p
            if i in range(0, nails_count // 4):
                drawer.text((x - font_cor_w, y - font_cor_h), f"{nails_count // 4 - i}", font=font, fill=ABCD['A'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(nails_count // 4, nails_count // 2):
                drawer.text((x - font_cor_w, y - font_cor_h), f"{nails_count // 2 - i}", font=font, fill=ABCD['D'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range(nails_count // 2, (nails_count // 4) * 3):
                drawer.text((x - font_cor_w, y - font_cor_h), f"{abs(i - (nails_count // 4) * 3)}", font=font,
                            fill=ABCD['C'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)
            if i in range((nails_count // 4) * 3, nails_count):
                drawer.text((x - font_cor_w, y - font_cor_h), f"{abs(i - nails_count)}", font=font, fill=ABCD['B'][1])
                drawer.line((xn, yn, x, y), fill='gray', width=1)

    for letter in ABCD:
        x, y = ABCD[letter][0][0] - ABCD_font_size // 3.1, ABCD[letter][0][1] - ABCD_font_size // 1.8
        drawer.text((x, y), letter, font=ImageFont.truetype("arial.ttf", ABCD_font_size), fill=ABCD[letter][1])

    blank = blank.convert('RGB')
    blank.save(file_path)


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

    # img = convert(args.input_file)
    img = mpimg.imread(args.input_file)
    if np.any(img > 100):
        img = img / 255

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

    print(f"Nails amount: {len(nails)}")

    if args.rgb:
        # ВАЖНО: wb=True => белая канва => линии должны ЗАТЕМНЯТЬ изображение
        iteration_strength = -0.1 if args.wb else 0.1

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        total_for_progress = (args.pull_amount or 0) * 3 if args.pull_amount else None

        # канвы под каналы: белая при wb=True
        str_pic_r = init_canvas(shape, black=not args.wb)
        pull_orders_r = create_art(nails, r, str_pic_r, iteration_strength,
                                   i_limit=args.pull_amount,
                                   progress_offset=0,
                                   progress_total=total_for_progress)

        str_pic_g = init_canvas(shape, black=not args.wb)
        pull_orders_g = create_art(nails, g, str_pic_g, iteration_strength,
                                   i_limit=args.pull_amount,
                                   progress_offset=(args.pull_amount or 0),
                                   progress_total=total_for_progress)

        str_pic_b = init_canvas(shape, black=not args.wb)
        pull_orders_b = create_art(nails, b, str_pic_b, iteration_strength,
                                   i_limit=args.pull_amount,
                                   progress_offset=(args.pull_amount or 0) * 2,
                                   progress_total=total_for_progress)

        max_pulls = np.max([len(pull_orders_r), len(pull_orders_g), len(pull_orders_b)])
        pull_orders_r = pull_orders_r + [pull_orders_r[-1]] * (max_pulls - len(pull_orders_r))
        pull_orders_g = pull_orders_g + [pull_orders_g[-1]] * (max_pulls - len(pull_orders_g))
        pull_orders_b = pull_orders_b + [pull_orders_b[-1]] * (max_pulls - len(pull_orders_b))

        pull_orders = [pull_orders_r, pull_orders_g, pull_orders_b]

        color_image_dimens = int(args.side_len * args.radius1_multiplier), int(
            args.side_len * args.radius2_multiplier), 3
        blank = init_canvas(color_image_dimens, black=not args.wb)  # белая/чёрная конечная канва

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
            -args.export_strength if args.wb else args.export_strength,  # ВАЖНО: wb True -> отрицательная сила
            args.rect
        )

        colors = {
            '255,0,255': 'Фиолетовый',
            '0,255,255': 'Голубой',
            '255,255,0': 'Желтый',
            '0,0,255': 'Синий',
            '0,0,0': 'Черный',
            '255,0,0': 'Красный',
            '0,255,0': 'Зеленый',
            '-1,-1,-1': 'Нет цвета'
        }

        df = pd.DataFrame([[pair[0], colors[pair[1]], pair[1]] for pair in instruction],
                          columns=['Нить', 'Цвет', 'RGB'])
        df.to_excel(f"{args.output_file}_instruction.xlsx", index=False)
        mpimg.imsave(args.output_file, result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)
        add_fields(args.output_file, args.rect, scaled_nails)
        print("100%", flush=True)

    else:
        orig_pic = rgb2gray(img) * 0.9

        image_dimens = int(args.side_len * args.radius1_multiplier), int(args.side_len * args.radius2_multiplier)

        # канва для моно: белая при wb=True
        if args.wb:
            str_pic = init_canvas(shape, black=False)
            pull_order = create_art(nails, orig_pic, str_pic, -0.05, i_limit=args.pull_amount,
                                    progress_offset=0, progress_total=args.pull_amount)
            blank = init_canvas(image_dimens, black=False)
            strength = -args.export_strength
        else:
            str_pic = init_canvas(shape, black=True)
            pull_order = create_art(nails, orig_pic, str_pic, 0.05, i_limit=args.pull_amount,
                                    progress_offset=0, progress_total=args.pull_amount)
            blank = init_canvas(image_dimens, black=True)
            strength = args.export_strength

        scaled_nails = scale_nails(
            image_dimens[1] / shape[1],
            image_dimens[0] / shape[0],
            nails
        )

        result = pull_order_to_array_bw(pull_order, blank, scaled_nails, strength)
        mpimg.imsave(args.output_file, result, cmap=plt.get_cmap("gray"), vmin=0.0, vmax=1.0)
        print("100%", flush=True)