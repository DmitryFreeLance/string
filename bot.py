import asyncio
import logging
import os
import uuid
import tempfile
from glob import glob
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, FSInputFile, InlineKeyboardMarkup,
    InlineKeyboardButton, CallbackQuery
)
from PIL import Image

# ================== НАСТРОЙКИ ==================
BOT_TOKEN = "7791601838:AAGKBsubpH1TzLYafINnCwz315Lf1qvkjxU"  # <-- поставь свой токен
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRINGART_SCRIPT = os.path.join(BASE_DIR, "stringart", "generate.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

NAIL_STEP_FOR_340 = "3"
PREVIEW_MAX_SIDE = 1600
PREVIEW_JPEG_QUALITY = 90

# ================== ПРОСТАЯ "БАЗА" КОДОВ ==================
CODES_DB = {
    "DEMO-3": {"limit": 3, "used": 0, "bound_to": None},
    "VIP-10": {"limit": 10, "used": 0, "bound_to": None},
}
user_codes = {}
# user_results[uid][idx] = {"png": path_to_png, "xlsx": path_to_xlsx or None, "png_mtime": float}
user_results = {}


# ================== УТИЛИТЫ ==================
def make_preview_jpeg(src_png: str) -> str:
    img = Image.open(src_png).convert("RGB")
    w, h = img.size
    k = PREVIEW_MAX_SIDE / max(w, h) if max(w, h) > PREVIEW_MAX_SIDE else 1.0
    if k < 1.0:
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
    prev_path = src_png.rsplit(".", 1)[0] + "_preview.jpg"
    img.save(prev_path, "JPEG", quality=PREVIEW_JPEG_QUALITY, optimize=True)
    return prev_path


def get_code_status(code: str):
    rec = CODES_DB.get(code)
    if not rec:
        return None
    return {"limit": rec["limit"], "used": rec["used"], "left": max(rec["limit"] - rec["used"], 0)}


def bind_code_to_user(uid: int, code: str):
    rec = CODES_DB.get(code)
    if not rec:
        return False, "❌ Код не найден. Проверьте написание."
    if rec["bound_to"] is None or rec["bound_to"] == uid:
        rec["bound_to"] = uid
        user_codes[uid] = code
        return True, "✅ Код привязан."
    return False, "❌ Этот код уже привязан к другому аккаунту."


def dec_use(uid: int):
    code = user_codes.get(uid)
    if not code:
        return
    rec = CODES_DB.get(code)
    if not rec:
        return
    rec["used"] = min(rec["used"] + 1, rec["limit"])


def _find_instruction_near_png(png_path: str) -> Optional[str]:
    """
    Ищет файл инструкции рядом с png:
      - <png>.png_instruction.xlsx  (твой кейс)
      - <png>_instruction.xlsx
      - *instruction*.xlsx
    """
    folder = os.path.dirname(png_path)
    base_with_ext = os.path.basename(png_path)        # abc.png
    base_no_ext, _ = os.path.splitext(base_with_ext)  # abc

    # самые вероятные имена
    candidates = [
        os.path.join(folder, base_with_ext + "_instruction.xlsx"),  # abc.png_instruction.xlsx
        os.path.join(folder, base_no_ext + "_instruction.xlsx"),    # abc_instruction.xlsx
    ]
    # любые instruction рядом
    patterns = [
        os.path.join(folder, f"{base_no_ext}*instruction*.xlsx"),
        os.path.join(folder, f"{base_with_ext}*instruction*.xlsx"),
        os.path.join(folder, "*instruction*.xlsx"),
    ]
    for patt in patterns:
        candidates.extend(glob(patt))

    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)

    for path in seen:
        if os.path.exists(path):
            return path
    return None


def _find_recent_xlsx(output_dir: str, ref_time: float, window_sec: int = 600) -> Optional[str]:
    """
    Находит .xlsx в папке output, созданный/изменённый максимально близко к ref_time.
    Если ничего не в окне — вернёт самый свежий .xlsx.
    """
    xlsx_files = glob(os.path.join(output_dir, "*.xlsx"))
    if not xlsx_files:
        return None

    best = None
    best_dt = None
    for path in xlsx_files:
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            continue
        dt = abs(mtime - ref_time)
        if dt <= window_sec:
            if best is None or dt < best_dt:
                best = path
                best_dt = dt

    if not best:
        try:
            best = max(xlsx_files, key=lambda p: os.path.getmtime(p))
        except Exception:
            best = None
    return best


# ================== КЛАВИАТУРЫ ==================
def kb_more_status(uid: int):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📤 Загрузить ещё фото", callback_data=f"more_{uid}"),
        InlineKeyboardButton(text="ℹ️ Мой статус", callback_data=f"status_{uid}")
    ]])


def kb_instruction(uid: int, idx: int):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📊 Получить инструкцию (Excel)", callback_data=f"choose_{uid}_{idx}")
    ]])


# ================== КОМАНДЫ ==================
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "✨ StringArt мастерская\n\n"
        "Я сделаю три цветных варианта картины на белом фоне:\n"
        "• ≈340 гвоздей, 4500 нитей\n"
        "• ≈340 гвоздей, 5000 нитей\n"
        "• ≈340 гвоздей, 5500 нитей\n\n"
        "Пришлите изображение одним сообщением. После обработки выберите вариант — пришлю инструкцию в Excel.\n\n"
        "Команда /status — привязать код и посмотреть остаток."
    )


@dp.message(Command("status"))
async def cmd_status(message: Message):
    uid = message.from_user.id
    args = message.text.strip().split(maxsplit=1)
    if len(args) == 2:
        code = args[1].strip()
        ok, msg = bind_code_to_user(uid, code)
        if not ok:
            await message.answer(msg)
            return

    code = user_codes.get(uid)
    if not code:
        await message.answer("ℹ️ Код не привязан. Отправьте /status ВАШ_КОД")
        return

    st = get_code_status(code)
    if not st:
        await message.answer("❌ Код не найден.")
        return

    await message.answer(
        f"🔐 Код: {code}\n"
        f"✅ Доступно: {st['left']} из {st['limit']}\n"
        f"🕒 Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    )


# ================== ОБРАБОТКА ФОТО ==================
@dp.message(F.photo)
async def handle_photo(message: Message):
    uid = message.from_user.id

    code = user_codes.get(uid)
    if code:
        st = get_code_status(code)
        if st and st["left"] <= 0:
            await message.answer("⚠️ По вашему коду взаимодействия закончились. Отправьте новый код через /status НОВЫЙ_КОД.")

    # сохраняем фото во временный файл
    photo = message.photo[-1]
    input_fd, input_path = tempfile.mkstemp(suffix=".jpg")
    os.close(input_fd)
    await bot.download(photo, destination=input_path)

    configs = [
        {"pull_amount": "500", "label": "Вариант 1 — ≈340 гвоздей, 4500 нитей"},
        {"pull_amount": "500", "label": "Вариант 2 — ≈340 гвоздей, 5000 нитей"},
        {"pull_amount": "500", "label": "Вариант 3 — ≈340 гвоздей, 5500 нитей"},
    ]

    results = {}
    all_ok = True

    for idx, cfg in enumerate(configs, start=1):
        output_png = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.png")
        expected_xlsx = f"{output_png}_instruction.xlsx"

        cmd = [
            "python", STRINGART_SCRIPT,
            "-i", input_path,
            "-o", output_png,
            "-d", "3000",
            "-s", "1",
            "-n", NAIL_STEP_FOR_340,
            "-l", cfg["pull_amount"],
            "-longside", "385",
            "--rgb",
            "--wb"
        ]

        progress_msg = await message.answer(f"⏳ {cfg['label']}: 0%")

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # читаем stdout построчно — ждём строки вида "NN%"
        last_progress = None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode(errors="ignore").strip()
            if text.endswith("%") and text != last_progress:
                last_progress = text
                try:
                    await progress_msg.edit_text(f"⏳ {cfg['label']}: {text}")
                except Exception:
                    pass

        await process.wait()
        stderr = (await process.stderr.read()).decode(errors="ignore")

        if process.returncode == 0 and os.path.exists(output_png):
            try:
                await progress_msg.edit_text(f"✅ {cfg['label']}: готово")
            except Exception:
                pass

            # превью
            preview_path = None
            try:
                preview_path = make_preview_jpeg(output_png)
                photo_file = FSInputFile(preview_path)
                await message.answer_photo(photo_file, caption=cfg["label"], reply_markup=kb_instruction(uid, idx))
            except Exception:
                try:
                    doc = FSInputFile(output_png)
                    await message.answer_document(doc, caption=cfg["label"], reply_markup=kb_instruction(uid, idx))
                except Exception as e:
                    await message.answer(f"❌ Ошибка при отправке результата «{cfg['label']}»:\n{e}")

            # --- поиск инструкции ---
            xlsx_path: Optional[str] = None

            if os.path.exists(expected_xlsx):
                xlsx_path = expected_xlsx

            if not xlsx_path:
                xlsx_path = _find_instruction_near_png(output_png)

            if not xlsx_path:
                try:
                    png_mtime = os.path.getmtime(output_png)
                except Exception:
                    png_mtime = datetime.now().timestamp()
                xlsx_path = _find_recent_xlsx(OUTPUT_DIR, ref_time=png_mtime, window_sec=600)

            results[str(idx)] = {
                "png": output_png,
                "xlsx": xlsx_path,
                "png_mtime": os.path.getmtime(output_png)
            }

            if preview_path and os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                except:
                    pass

            if not xlsx_path:
                await message.answer(
                    "⚠️ Инструкция Excel пока не найдена. Нажмите «📊 Получить инструкцию» — я проверю ещё раз."
                )
        else:
            all_ok = False
            try:
                await progress_msg.edit_text(f"❌ {cfg['label']}: ошибка генерации")
            except Exception:
                pass
            await message.answer(f"❌ Ошибка при генерации «{cfg['label']}»:\n{stderr}")
            if os.path.exists(output_png):
                try:
                    os.remove(output_png)
                except:
                    pass

    user_results[uid] = results
    dec_use(uid)

    if all_ok:
        await message.answer(
            "🎉 Все три варианта готовы! Выберите понравившийся и нажмите «📊 Получить инструкцию».\n\n"
            "Хотите попробовать другое фото? Нажмите «Загрузить ещё фото» ниже ⬇️",
            reply_markup=kb_more_status(uid)
        )
    else:
        await message.answer(
            "Готово с предупреждениями. Можно отправить другое фото или запросить инструкцию для удачных вариантов.",
            reply_markup=kb_more_status(uid)
        )

    # чистим входной файл
    if os.path.exists(input_path):
        try:
            os.remove(input_path)
        except:
            pass


# ================== КНОПКИ ==================
@dp.callback_query(F.data.startswith("choose_"))
async def handle_choice(callback: CallbackQuery):
    _, uid, idx = callback.data.split("_")
    uid = int(uid)

    if uid not in user_results or idx not in user_results[uid]:
        await callback.answer("Результат не найден", show_alert=True)
        return

    files = user_results[uid][idx]
    xlsx_path = files.get("xlsx")
    png_path = files.get("png")
    png_mtime = files.get("png_mtime") or (os.path.getmtime(png_path) if png_path and os.path.exists(png_path) else None)

    # повторный поиск, если путь пустой/устарел
    if (not xlsx_path) or (xlsx_path and not os.path.exists(xlsx_path)):
        # 1) рядом с PNG
        if png_path and os.path.exists(png_path):
            xlsx_path = _find_instruction_near_png(png_path)
        # 2) по времени в OUTPUT_DIR
        if (not xlsx_path) and png_mtime:
            xlsx_path = _find_recent_xlsx(OUTPUT_DIR, ref_time=png_mtime, window_sec=600)
        files["xlsx"] = xlsx_path  # обновим кеш

    if xlsx_path and os.path.exists(xlsx_path):
        try:
            doc = FSInputFile(xlsx_path, filename=os.path.basename(xlsx_path))
            await callback.message.answer_document(doc, caption="📊 Ваша инструкция (Excel)")
        except Exception as e:
            await callback.message.answer(f"❌ Не удалось отправить файл:\n{e}")
        # можно удалить файл после отправки
        try:
            os.remove(xlsx_path)
        except:
            pass
    else:
        # отладочная подсказка
        folder = os.path.dirname(png_path) if png_path else OUTPUT_DIR
        nearby = "\n".join(os.path.basename(p) for p in glob(os.path.join(folder, "*instruction*.xlsx")))
        await callback.message.answer(
            "❌ Инструкция не найдена.\n\n"
            f"Искал рядом с PNG и в {OUTPUT_DIR}.\n"
            f"PNG: {png_path}\n"
            f"Папка: {folder}\n"
            f"Найдено рядом: {nearby or 'ничего'}"
        )

    # PNG очищаем опционально
    if png_path and os.path.exists(png_path):
        try:
            os.remove(png_path)
        except:
            pass

    # чистим запись
    try:
        del user_results[uid][idx]
        if not user_results[uid]:
            del user_results[uid]
    except:
        pass

    await callback.answer()


@dp.callback_query(F.data.startswith("more_"))
async def handle_more(callback: CallbackQuery):
    await callback.message.answer("📸 Пришлите новое фото одним сообщением — я снова сделаю три варианта!")
    await callback.answer()


@dp.callback_query(F.data.startswith("status_"))
async def handle_inline_status(callback: CallbackQuery):
    uid = int(callback.data.split("_")[1])
    code = user_codes.get(uid)
    if not code:
        await callback.message.answer("ℹ️ Код не привязан. Отправьте /status ВАШ_КОД")
        await callback.answer()
        return
    st = get_code_status(code)
    if not st:
        await callback.message.answer("❌ Код не найден.")
    else:
        await callback.message.answer(f"🔐 Код: {code}\n✅ Доступно: {st['left']} из {st['limit']}")
    await callback.answer()


# ================== ЗАПУСК ==================
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())