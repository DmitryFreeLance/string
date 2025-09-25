from __future__ import annotations

import asyncio
import logging
import os
import uuid
import tempfile
import json
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, FSInputFile, InlineKeyboardMarkup,
    InlineKeyboardButton, CallbackQuery, ContentType
)
from PIL import Image

# ==== Настройки ====
BOT_TOKEN = "7791601838:AAGKBsubpH1TzLYafINnCwz315Lf1qvkjxU"  # <--- замени
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRINGART_SCRIPT = os.path.join(BASE_DIR, "stringart", "generate.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Админы (кто может добавлять коды)
ADMIN_IDS = {726773708}  # <--- Вставь свой telegram user_id (int)

# Превью для Telegram
PREVIEW_MAX_SIDE = 1600
PREVIEW_JPEG_QUALITY = 90

# Гвозди/параметры
NAIL_STEP_FOR_340 = "3"

# ===== Персистентная БД кодов =====
CODES_FILE = os.path.join(BASE_DIR, "codes.json")

def load_codes_db() -> dict:
    if os.path.exists(CODES_FILE):
        try:
            with open(CODES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # ожидаем формат: { CODE: {"limit": int, "used_images": int, "bound_to": int|None} }
                return data
        except Exception:
            pass
    # дефолт (можно убрать)
    return {
        "DEMO-5": {"limit": 5, "used_images": 0, "bound_to": None},
    }

def save_codes_db(data: dict) -> None:
    with open(CODES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

CODES_DB = load_codes_db()

# user_codes[user_id] = "CODE"
user_codes: dict[int, str] = {}

# user_results[uid][idx] = {"png": path, "xlsx": path}
user_results: dict[int, dict[str, dict]] = {}

# Кто сейчас генерирует (чтобы блокировать новые фото/команды)
user_busy: set[int] = set()

# ===== Утилиты кодов =====
def get_code_status(code: str):
    rec = CODES_DB.get(code)
    if not rec:
        return None
    left = max(rec["limit"] - rec.get("used_images", 0), 0)
    return {"limit": rec["limit"], "used": rec.get("used_images", 0), "left": left, "bound_to": rec.get("bound_to")}

def bind_code_to_user(uid: int, code: str):
    rec = CODES_DB.get(code)
    if not rec:
        return False, "Код не найден."
    if rec.get("bound_to") is None or rec.get("bound_to") == uid:
        rec["bound_to"] = uid
        CODES_DB[code] = rec
        user_codes[uid] = code
        save_codes_db(CODES_DB)
        return True, "Код привязан."
    return False, "Этот код уже привязан к другому аккаунту."

def can_use_code(uid: int) -> tuple[bool, str | None]:
    code = user_codes.get(uid)
    if not code:
        return False, "Код не привязан. Отправьте: /status ВАШ_КОД"
    st = get_code_status(code)
    if not st:
        return False, "Код не найден."
    if st["left"] <= 0:
        return False, "По вашему коду лимит изображений исчерпан. Отправьте новый код: /status НОВЫЙ_КОД"
    return True, None

def inc_code_usage(uid: int):
    code = user_codes.get(uid)
    if not code:
        return
    rec = CODES_DB.get(code)
    if not rec:
        return
    rec["used_images"] = rec.get("used_images", 0) + 1
    CODES_DB[code] = rec
    save_codes_db(CODES_DB)

# ===== Работа с превью =====
def make_preview_jpeg(src_png: str) -> str:
    img = Image.open(src_png).convert("RGB")
    w, h = img.size
    k = PREVIEW_MAX_SIDE / max(w, h) if max(w, h) > PREVIEW_MAX_SIDE else 1.0
    if k < 1.0:
        img = img.resize((int(w * k), int(h * k)), Image.LANCZOS)
    prev_path = src_png.rsplit(".", 1)[0] + "_preview.jpg"
    img.save(prev_path, "JPEG", quality=PREVIEW_JPEG_QUALITY, optimize=True)
    return prev_path

# ===== Клавиатуры =====
def kb_more_status(uid: int):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📤 Загрузить ещё фото", callback_data=f"more_{uid}"),
        InlineKeyboardButton(text="ℹ️ Мой статус", callback_data=f"status_{uid}")
    ]])

def kb_instruction(uid: int, idx: int):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📊 Получить инструкцию (Excel)", callback_data=f"choose_{uid}_{idx}")
    ]])

# ===== Импорт кодов админом =====
def parse_codes_from_text(text: str) -> list[str]:
    text = text.replace("；", ";").replace("、", ",").replace("—", "-")
    raw_parts = [p.strip() for p in text.replace(";", ",").split(",")]
    parts = [p for p in raw_parts if p]
    return parts

def add_codes_to_db(parts: list[str], default_limit: int = 5) -> tuple[int, int]:
    added, updated = 0, 0
    for part in parts:
        code = part
        limit = default_limit
        if ":" in part:
            code, lim = part.split(":", 1)
            code = code.strip()
            try:
                limit = max(0, int(lim.strip()))
            except:
                limit = default_limit
        if not code:
            continue
        if code in CODES_DB:
            if CODES_DB[code].get("limit") != limit:
                CODES_DB[code]["limit"] = limit
                updated += 1
        else:
            CODES_DB[code] = {"limit": limit, "used_images": 0, "bound_to": None}
            added += 1
    save_codes_db(CODES_DB)
    return added, updated

# ===== Команды =====
@dp.message(Command("start"))
async def cmd_start(message: Message):
    uid = message.from_user.id
    if uid in user_busy:
        await message.answer("Сейчас идёт генерация. Подождите завершения, пожалуйста.")
        return

    hello = (
        "✨ StringArt мастерская\n\n"
        "Пришлите фото, я сделаю 3 цветных варианта на белом фоне:\n"
        "• 360 гвоздей, 4500 нитей\n"
        "• 360 гвоздей, 5000 нитей\n"
        "• 360 гвоздей, 5500 нитей\n\n"
        "Затем по кнопке пришлю инструкцию в Excel.\n\n"
        "Команда: /status — привязать код и узнать остаток.\n"
        "(По одному коду можно обработать до 5 изображений.)"
    )
    await message.answer(hello)

@dp.message(Command("status"))
async def cmd_status(message: Message):
    uid = message.from_user.id
    if uid in user_busy:
        await message.answer("Сейчас идёт генерация. Подождите завершения, пожалуйста.")
        return

    args = message.text.strip().split(maxsplit=1)
    if len(args) == 2:
        code = args[1].strip()
        ok, msg = bind_code_to_user(uid, code)
        if not ok:
            await message.answer("❌ " + msg)
            return
        await message.answer("✅ " + msg)

    code = user_codes.get(uid)
    if not code:
        await message.answer("Код не привязан. Отправьте `/status ВАШ_КОД`", parse_mode=None)
        return

    st = get_code_status(code)
    if not st:
        await message.answer("Код не найден.")
        return

    await message.answer(
        f"Код: {code}\n"
        f"Доступно: {st['left']} из {st['limit']}\n"
        f"Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    )

@dp.message(Command("add"))
async def cmd_add_codes(message: Message):
    uid = message.from_user.id
    if uid not in ADMIN_IDS:
        await message.answer("⛔️ Команда только для админов.")
        return
    if uid in user_busy:
        await message.answer("Сейчас идёт генерация (у вас). Дождитесь завершения.")
        return

    parts_raw = message.text.split(maxsplit=1)
    if len(parts_raw) < 2:
        await message.answer("Формат: /add CODE1,CODE2 или /add CODE1:5,CODE2:10")
        return

    parts = parse_codes_from_text(parts_raw[1])
    if not parts:
        await message.answer("Не удалось распознать коды. Проверь формат.")
        return

    added, updated = add_codes_to_db(parts, default_limit=5)
    await message.answer(f"Импорт: добавлено {added}, обновлено {updated}. Всего кодов: {len(CODES_DB)}")

# ===== Обработка документов админом (docx/txt/csv) =====
@dp.message(F.document)
async def handle_doc_codes(message: Message):
    uid = message.from_user.id
    if uid not in ADMIN_IDS:
        return
    if uid in user_busy:
        await message.answer("Сейчас идёт генерация (у вас). Дождитесь завершения.")
        return

    from docx import Document  # локальный импорт, если docx не установлен — отвалится только тут

    doc = message.document
    file_name = (doc.file_name or "").lower()
    allowed = file_name.endswith(".docx") or file_name.endswith(".txt") or file_name.endswith(".csv")
    if not allowed:
        await message.answer("Пришлите .docx, .txt или .csv с кодами через запятую.")
        return

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(file_name)[1])
    os.close(tmp_fd)
    try:
        await bot.download(doc, destination=tmp_path)

        text = ""
        if file_name.endswith(".docx"):
            try:
                d = Document(tmp_path)
                text = "\n".join(p.text for p in d.paragraphs)
            except Exception as e:
                await message.answer(f"Не удалось прочитать .docx: {e}")
                return
        else:
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(tmp_path, "r", encoding="cp1251") as f:
                    text = f.read()

        if not text.strip():
            await message.answer("Файл пустой или не удалось прочитать содержимое.")
            return

        parts = parse_codes_from_text(text)
        if not parts:
            await message.answer("Не нашёл кодов. Формат: CODE1,CODE2 или CODE:LIMIT.")
            return

        added, updated = add_codes_to_db(parts, default_limit=5)
        await message.answer(f"Импорт завершён. Добавлено: {added}, обновлено: {updated}, всего: {len(CODES_DB)}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

# ===== Генерация по фото =====
@dp.message(F.photo)
async def handle_photo(message: Message):
    uid = message.from_user.id

    # Блок в процессе
    if uid in user_busy:
        await message.answer("Сейчас идёт генерация. Дождитесь завершения, пожалуйста.")
        return

    # Проверка кода и лимита
    ok, err = can_use_code(uid)
    if not ok:
        await message.answer("❌ " + err)
        return

    # Сохраняем входное фото
    photo = message.photo[-1]
    input_fd, input_path = tempfile.mkstemp(suffix=".jpg")
    os.close(input_fd)
    await bot.download(photo, destination=input_path)

    # Регистрируем результаты
    if uid not in user_results:
        user_results[uid] = {}

    # Помечаем «занят»
    user_busy.add(uid)

    # Списываем 1 использование (одно фото = 1 списание)
    inc_code_usage(uid)

    try:
        configs = [
            {"pull_amount": "1500", "label": "Вариант 1 — 360 гвоздей, 4500 нитей"},
            {"pull_amount": "2000", "label": "Вариант 2 — 360 гвоздей, 5000 нитей"},
            {"pull_amount": "2500", "label": "Вариант 3 — 360 гвоздей, 5500 нитей"},
        ]

        for idx, cfg in enumerate(configs, start=1):
            output_png = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.png")
            output_xlsx = f"{output_png}_instruction.xlsx"

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
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            last_progress = None
            # читаем stdout для процентов
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

                # положим результат для кнопки Excel ДО отправки превью
                user_results.setdefault(uid, {})
                user_results[uid][str(idx)] = {"png": output_png, "xlsx": output_xlsx}

                # Отправим превью/док сразу по готовности варианта
                try:
                    preview_path = make_preview_jpeg(output_png)
                    photo_file = FSInputFile(preview_path)
                    await message.answer_photo(
                        photo_file,
                        caption=cfg["label"],
                        reply_markup=kb_instruction(uid, idx)
                    )
                    try:
                        os.remove(preview_path)
                    except:
                        pass
                except Exception:
                    # резерв: шлём как документ
                    try:
                        doc = FSInputFile(output_png)
                        await message.answer_document(
                            doc, caption=cfg["label"], reply_markup=kb_instruction(uid, idx)
                        )
                    except Exception as e:
                        await message.answer(f"Ошибка отправки результата: {e}")

                # Сообщим, если Excel не создан
                if not os.path.exists(output_xlsx):
                    await message.answer(
                        "⚠️ Инструкция Excel для этого варианта пока не найдена. "
                        "Если она нужна — попробуй нажать кнопку позже."
                    )
            else:
                try:
                    await progress_msg.edit_text(f"❌ {cfg['label']}: ошибка генерации")
                except Exception:
                    pass
                await message.answer(f"Ошибка генерации «{cfg['label']}»:\n{stderr or 'нет лога'}")

        # Финальное сообщение
        await message.answer(
            "Готово! Можешь получить инструкции по кнопкам у каждого варианта.\n"
            "Чтобы попробовать другое фото — нажми «Загрузить ещё фото».",
            reply_markup=kb_more_status(uid)
        )

    finally:
        # Разблокируем
        user_busy.discard(uid)
        # Чистим вход
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass

# ===== Кнопки =====
@dp.callback_query(F.data.startswith("choose_"))
async def handle_choice(callback: CallbackQuery):
    try:
        _, uid, idx = callback.data.split("_")
        uid = int(uid)
    except Exception:
        await callback.answer("Ошибка данных", show_alert=True)
        return

    # Получение Excel разрешено даже если user_busy
    if uid not in user_results or idx not in user_results[uid]:
        await callback.answer("Результат не найден", show_alert=True)
        return

    files = user_results[uid][idx]
    xlsx_path = files.get("xlsx")
    png_path = files.get("png")

    if xlsx_path and os.path.exists(xlsx_path):
        doc = FSInputFile(xlsx_path)
        await callback.message.answer_document(doc, caption="📊 Ваша инструкция (Excel)")
        # по желанию — удалить после отправки:
        try:
            os.remove(xlsx_path)
        except:
            pass
    else:
        await callback.message.answer("❌ Инструкция не найдена. Попробуйте чуть позже.")

    # Можно подчистить PNG, если не нужен
    if png_path and os.path.exists(png_path):
        try:
            os.remove(png_path)
        except:
            pass

    # Убираем запись для выбранного варианта
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
    if uid in user_busy:
        await callback.message.answer("Сейчас идёт генерация. Дождитесь завершения.")
    else:
        await callback.message.answer("📸 Пришлите новое фото одним сообщением — сделаю три варианта!")
    await callback.answer()

@dp.callback_query(F.data.startswith("status_"))
async def handle_inline_status(callback: CallbackQuery):
    uid = int(callback.data.split("_")[1])
    code = user_codes.get(uid)
    if not code:
        await callback.message.answer("Код не привязан. Отправьте `/status ВАШ_КОД`")
        await callback.answer()
        return
    st = get_code_status(code)
    if not st:
        await callback.message.answer("Код не найден.")
    else:
        await callback.message.answer(f"Код: {code}\nДоступно: {st['left']} из {st['limit']}")
    await callback.answer()

# Блокируем команды во время генерации (кроме /add у админов и /status без кода?)
@dp.message(F.text & ~F.photo & ~F.document)
async def generic_text(message: Message):
    uid = message.from_user.id
    text = (message.text or "").strip()
    if text.startswith("/"):
        # команды закрываем во время генерации
        if uid in user_busy:
            await message.answer("Сейчас идёт генерация. Дождитесь завершения, пожалуйста.")
            return
    # Ничего не делаем — остальные обработчики уже покрыты

# ===== Запуск =====
async def main():
    # Сброс webhook (на всякий)
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())