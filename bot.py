import asyncio
import logging
import os
import uuid
import tempfile
from glob import glob
from datetime import datetime
from typing import Optional, Dict, Any, Set

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, FSInputFile, InlineKeyboardMarkup,
    InlineKeyboardButton, CallbackQuery
)
from PIL import Image

# ================== НАСТРОЙКИ ==================
BOT_TOKEN = "7791601838:AAGKBsubpH1TzLYafINnCwz315Lf1qvkjxU"   # <-- ПОСТАВЬ СВОЙ ТОКЕН
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

# Сколько изображений можно обработать по одному коду
IMAGES_LIMIT_PER_CODE = 5

# ================== ПРОСТАЯ "БАЗА" КОДОВ ==================
# структура кода:
# CODES_DB[code] = {"images_limit": int, "images_used": int, "bound_to": Optional[user_id]}
CODES_DB: Dict[str, Dict[str, Any]] = {
    "DEMO-5": {"images_limit": 5, "images_used": 0, "bound_to": None},
    "VIP-5": {"images_limit": 5, "images_used": 0, "bound_to": None},
}

# user_codes[user_id] = bound_code
user_codes: Dict[int, str] = {}
# user_results[uid][idx] = {"png": path_to_png, "xlsx": path_to_xlsx or None, "png_mtime": float}
user_results: Dict[int, Dict[str, Dict[str, Any]]] = {}
# Пользователи, у которых сейчас идёт генерация (блокируем команды и новые фото)
busy_users: Set[int] = set()


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


def get_code_status(code: str) -> Optional[Dict[str, int]]:
    rec = CODES_DB.get(code)
    if not rec:
        return None
    left = max(rec["images_limit"] - rec["images_used"], 0)
    return {"limit": rec["images_limit"], "used": rec["images_used"], "left": left}


def bind_code_to_user(uid: int, code: str):
    rec = CODES_DB.get(code)
    if not rec:
        return False, "❌ Код не найден. Проверьте написание."
    if rec["bound_to"] is None or rec["bound_to"] == uid:
        rec["bound_to"] = uid
        user_codes[uid] = code
        return True, "✅ Код привязан."
    return False, "❌ Этот код уже привязан к другому аккаунту."


def dec_image_use(uid: int):
    """Списать 1 изображение (сессию из 3 вариантов) с привязанного кода."""
    code = user_codes.get(uid)
    if not code:
        return
    rec = CODES_DB.get(code)
    if not rec:
        return
    rec["images_used"] = min(rec["images_used"] + 1, rec["images_limit"])


def _find_instruction_near_png(png_path: str) -> Optional[str]:
    """
    Ищет файл инструкции рядом с png:
      - <png>.png_instruction.xlsx
      - <png>_instruction.xlsx
      - *instruction*.xlsx
    """
    folder = os.path.dirname(png_path)
    base_with_ext = os.path.basename(png_path)        # abc.png
    base_no_ext, _ = os.path.splitext(base_with_ext)  # abc

    candidates = [
        os.path.join(folder, base_with_ext + "_instruction.xlsx"),
        os.path.join(folder, base_no_ext + "_instruction.xlsx"),
    ]
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
    """Находит .xlsx в папке output, созданный/изменённый максимально близко к ref_time."""
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


# ================== ХЕЛПЕР “занятости” ==================
def is_busy(uid: int) -> bool:
    return uid in busy_users

def require_not_busy(func):
    """
    Блокируем команды/фото, если у пользователя идёт генерация.
    ВАЖНО: не пробрасываем **kwargs в сам хендлер, чтобы не было
    TypeError: ... unexpected keyword argument 'dispatcher'
    """
    async def wrapper(message: Message, *args, **kwargs):
        uid = message.from_user.id
        if is_busy(uid):
            await message.answer("⏳ Идёт генерация. Подождите завершения, пожалуйста. Excel можно получать по кнопке.")
            return
        # хендлеры у тебя имеют сигнатуру только (message: Message),
        # поэтому передаём только message и НИЧЕГО из kwargs
        return await func(message)
    return wrapper
# ================== КОМАНДЫ ==================
@dp.message(Command("start"))
@require_not_busy
async def cmd_start(message: Message):
    await message.answer(
        "✨ StringArt мастерская\n\n"
        "Я сделаю три цветных варианта картины на белом фоне:\n"
        "• ≈340 гвоздей, 4500 нитей\n"
        "• ≈340 гвоздей, 5000 нитей\n"
        "• ≈340 гвоздей, 5500 нитей\n\n"
        "Пришлите изображение одним сообщением. После обработки выберите вариант — пришлю инструкцию в Excel.\n\n"
        "Команда /status — привязать код и посмотреть остаток.\n"
        "Команда /add код1,код2,... — массово добавить коды (лимит 5 изображений на код)."
    )


@dp.message(Command("status"))
@require_not_busy
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
        f"📦 Лимит изображений: {st['limit']}\n"
        f"🧮 Израсходовано: {st['used']}\n"
        f"✅ Осталось: {st['left']}\n"
        f"🕒 Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    )


@dp.message(Command("add"))
@require_not_busy
async def cmd_add(message: Message):
    """
    /add code1,code2,code3
    добавляет коды в базу с лимитом IMAGES_LIMIT_PER_CODE на каждый.
    """
    text = message.text.strip()
    parts = text.split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        await message.answer("Использование: /add код1,код2,код3")
        return

    raw = parts[1]
    tokens = [t.strip() for t in raw.replace("\n", ",").split(",")]
    tokens = [t for t in tokens if t]

    if not tokens:
        await message.answer("Не нашёл кодов в команде. Пример: /add ABC123, XYZ-999")
        return

    added = 0
    skipped = 0
    for code in tokens:
        if code in CODES_DB:
            skipped += 1
            continue
        CODES_DB[code] = {
            "images_limit": IMAGES_LIMIT_PER_CODE,
            "images_used": 0,
            "bound_to": None
        }
        added += 1

    await message.answer(f"✅ Добавлено кодов: {added}\n↪️ Пропущено (уже есть): {skipped}\nЛимит на код: {IMAGES_LIMIT_PER_CODE} изображений.")


# ================== ОБРАБОТКА ФОТО ==================
@dp.message(F.photo)
@require_not_busy
async def handle_photo(message: Message):
    uid = message.from_user.id

    # требуется привязанный код
    code = user_codes.get(uid)
    if not code:
        await message.answer("🔐 Сначала привяжите код: /status ВАШ_КОД")
        return

    st = get_code_status(code)
    if not st:
        await message.answer("❌ Код не найден. Отправьте /status НОВЫЙ_КОД")
        return
    if st["left"] <= 0:
        await message.answer("⚠️ По вашему коду лимит изображений исчерпан. Отправьте новый код через /status НОВЫЙ_КОД.")
        return

    # помечаем пользователя занятым до завершения генерации
    busy_users.add(uid)

    # гарантируем хранение результатов
    user_results.setdefault(uid, {})

    try:
        # сохраняем фото во временный файл
        photo = message.photo[-1]
        input_fd, input_path = tempfile.mkstemp(suffix=".jpg")
        os.close(input_fd)
        await bot.download(photo, destination=input_path)

        configs = [
            {"pull_amount": "4500", "label": "Вариант 1 — ≈340 гвоздей, 4500 нитей"},
            {"pull_amount": "5000", "label": "Вариант 2 — ≈340 гвоздей, 5000 нитей"},
            {"pull_amount": "5500", "label": "Вариант 3 — ≈340 гвоздей, 5500 нитей"},
        ]

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

                # фиксируем mtime сразу
                try:
                    png_mtime = os.path.getmtime(output_png)
                except Exception:
                    png_mtime = datetime.now().timestamp()

                # находим/запоминаем Excel
                xlsx_path: Optional[str] = expected_xlsx if os.path.exists(expected_xlsx) else None
                if not xlsx_path:
                    xlsx_path = _find_instruction_near_png(output_png)
                if not xlsx_path:
                    xlsx_path = _find_recent_xlsx(OUTPUT_DIR, ref_time=png_mtime, window_sec=600)

                # сохраняем результат для кнопки
                user_results.setdefault(uid, {})
                user_results[uid][str(idx)] = {
                    "png": output_png,
                    "xlsx": xlsx_path,
                    "png_mtime": png_mtime
                }

                # отправляем превью + кнопку
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

                if preview_path and os.path.exists(preview_path):
                    try:
                        os.remove(preview_path)
                    except:
                        pass

                if not xlsx_path:
                    await message.answer(
                        f"⚠️ Инструкция для «{cfg['label']}» пока не найдена. "
                        f"Нажмите «📊 Получить инструкцию» — я проверю ещё раз."
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

        # списываем 1 изображение с кода (одна сессия из трёх вариантов = одно изображение)
        dec_image_use(uid)

        if all_ok:
            await message.answer(
                "🎉 Варианты поступают по мере готовности. Для каждого уже можно запрашивать Excel.\n\n"
                "Хотите попробовать другое фото? Нажмите «Загрузить ещё фото» ниже ⬇️",
                reply_markup=kb_more_status(uid)
            )
        else:
            await message.answer(
                "Готово с предупреждениями. Для успешно сгенерированных вариантов можно запросить Excel, либо отправить другое фото.",
                reply_markup=kb_more_status(uid)
            )

    finally:
        # снимаем “занято”
        if uid in busy_users:
            busy_users.remove(uid)

        # чистим входной файл
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass


# ================== КНОПКИ (Excel разрешён даже во время генерации) ==================
@dp.callback_query(F.data.startswith("choose_"))
async def handle_choice(callback: CallbackQuery):
    _, uid, idx = callback.data.split("_")
    uid = int(uid)

    # структура на всякий
    user_results.setdefault(uid, {})

    if idx not in user_results[uid]:
        await callback.answer("Результат не найден", show_alert=True)
        return

    files = user_results[uid][idx]
    xlsx_path = files.get("xlsx")
    png_path = files.get("png")
    png_mtime = files.get("png_mtime") or (os.path.getmtime(png_path) if png_path and os.path.exists(png_path) else None)

    # повторный поиск на случай, если файл появился только что
    if (not xlsx_path) or (xlsx_path and not os.path.exists(xlsx_path)):
        if png_path and os.path.exists(png_path):
            xlsx_path = _find_instruction_near_png(png_path)
        if (not xlsx_path) and png_mtime:
            xlsx_path = _find_recent_xlsx(OUTPUT_DIR, ref_time=png_mtime, window_sec=600)
        files["xlsx"] = xlsx_path

    if xlsx_path and os.path.exists(xlsx_path):
        try:
            doc = FSInputFile(xlsx_path, filename=os.path.basename(xlsx_path))
            await callback.message.answer_document(doc, caption="📊 Ваша инструкция (Excel)")
        except Exception as e:
            await callback.message.answer(f"❌ Не удалось отправить файл:\n{e}")
        # опционально чистим xlsx
        try:
            os.remove(xlsx_path)
        except:
            pass
    else:
        folder = os.path.dirname(png_path) if png_path else OUTPUT_DIR
        nearby = "\n".join(os.path.basename(p) for p in glob(os.path.join(folder, "*instruction*.xlsx")))
        await callback.message.answer(
            "❌ Инструкция пока не найдена.\n\n"
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
    uid = int(callback.data.split("_")[1])
    if is_busy(uid):
        await callback.message.answer("⏳ Сейчас идёт генерация. Дождитесь завершения, пожалуйста.")
    else:
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
        await callback.message.answer(
            f"🔐 Код: {code}\n"
            f"📦 Лимит изображений: {st['limit']}\n"
            f"🧮 Израсходовано: {st['used']}\n"
            f"✅ Осталось: {st['left']}"
        )
    await callback.answer()


# ================== ЗАПУСК ==================
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
