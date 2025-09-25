import asyncio
import logging
import os
import uuid
import tempfile
from datetime import datetime
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, FSInputFile, InlineKeyboardMarkup,
    InlineKeyboardButton, CallbackQuery
)
from PIL import Image  # для JPEG-превью

# ================== НАСТРОЙКИ ==================
BOT_TOKEN = "7791601838:AAGKBsubpH1TzLYafINnCwz315Lf1qvkjxU"  # 🔴 замени на свой токен
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRINGART_SCRIPT = os.path.join(BASE_DIR, "stringart", "generate.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# шаг по периметру, дающий ≈340 гвоздей на твоём поле
NAIL_STEP_FOR_340 = "3"

# размеры превью (чтобы sendPhoto не ронял соединение)
PREVIEW_MAX_SIDE = 1600
PREVIEW_JPEG_QUALITY = 90

# ================== ПСЕВДО-БАЗА КОДОВ ==================
# ❗️ Замени на свою БД. Здесь — только для примера.
# Каждая запись: {"limit": N, "used": 0, "bound_to": user_id or None}
CODES_DB = {
    "DEMO-3": {"limit": 3, "used": 0, "bound_to": None},
    "VIP-10": {"limit": 10, "used": 0, "bound_to": None},
}

# user_codes[user_id] = "CODE"
user_codes = {}

# user_results[uid][idx] = {"png": path_to_png, "xlsx": path_to_xlsx}
user_results = {}

# ================== УТИЛИТЫ ==================
def make_preview_jpeg(src_png: str) -> str:
    """Создаёт ужатое JPEG-превью для Telegram (<= ~1600px по большой стороне)."""
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
    """Привязывает код к пользователю, если свободен. Возвращает (ok: bool, msg: str)."""
    rec = CODES_DB.get(code)
    if not rec:
        return False, "❌ Код не найден. Проверьте написание."
    if rec["bound_to"] is None or rec["bound_to"] == uid:
        rec["bound_to"] = uid
        user_codes[uid] = code
        return True, "✅ Код привязан."
    return False, "❌ Этот код уже привязан к другому аккаунту."

def dec_use(uid: int):
    """Списать 1 взаимодействие с кода, если привязан."""
    code = user_codes.get(uid)
    if not code:
        return
    rec = CODES_DB.get(code)
    if not rec:
        return
    rec["used"] = min(rec["used"] + 1, rec["limit"])

# ================== КЛАВИАТУРЫ ==================
def kb_more_status(uid: int):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📤 Загрузить ещё фото", callback_data=f"more_{uid}"),
        InlineKeyboardButton(text="ℹ️ Мой статус", callback_data=f"status_{uid}")
    ]])

def kb_instruction(uid: int, idx: int):
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="📊 Получить инструкцию (Excel)",
            callback_data=f"choose_{uid}_{idx}"
        )
    ]])

# ================== КОМАНДЫ ==================
@dp.message(Command("start"))
async def cmd_start(message: Message):
    hello = (
        "✨ *StringArt мастерская* \n\n"
        "Я превращу ваше фото в три *цветных* варианта картины на *белом фоне*:\n"
        "• ≈340 гвоздей, 4500 нитей\n"
        "• ≈340 гвоздей, 5000 нитей\n"
        "• ≈340 гвоздей, 5500 нитей\n\n"
        "Просто пришлите изображение в чат. После обработки выберите понравившийся вариант — пришлю инструкцию в Excel 📊\n\n"
        "_Команда_/status — проверить/привязать код и узнать, сколько взаимодействий осталось."
    )
    await message.answer(hello, parse_mode="Markdown")

@dp.message(Command("status"))
async def cmd_status(message: Message):
    """
    /status                 -> показать статус по уже привязанному коду
    /status <CODE>          -> привязать код и показать статус
    """
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
        await message.answer("ℹ️ Код не привязан. Отправьте `/status ВАШ_КОД`", parse_mode="Markdown")
        return

    st = get_code_status(code)
    if not st:
        await message.answer("❌ Код не найден.")
        return

    await message.answer(
        f"🔐 Код: *{code}*\n"
        f"✅ Доступно: *{st['left']}* из *{st['limit']}*\n"
        f"🕒 Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
        parse_mode="Markdown"
    )

# ================== ОБРАБОТКА ФОТО ==================
@dp.message(F.photo)
async def handle_photo(message: Message):
    uid = message.from_user.id

    # Проверим привязку кода и остаток (не блокируем жёстко, но предупредим)
    code = user_codes.get(uid)
    if code:
        st = get_code_status(code)
        if st and st["left"] <= 0:
            await message.answer(
                "⚠️ По вашему коду взаимодействия закончились. Отправьте новый код через `/status НОВЫЙ_КОД`.",
                parse_mode="Markdown"
            )

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

    results = {}
    all_ok = True

    for idx, cfg in enumerate(configs, start=1):
        output_png = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.png")
        output_xlsx = f"{output_png}_instruction.xlsx"

        # Цветной + белый фон, «сочные» параметры (как у тебя):
        cmd = [
            "python", STRINGART_SCRIPT,
            "-i", input_path,
            "-o", output_png,
            "-d", "3000",                # dimension
            "-s", "1",                   # strength
            "-n", NAIL_STEP_FOR_340,     # ~340 гвоздей (шаг по периметру)
            "-l", cfg["pull_amount"],    # pulls
            "-longside", "385",          # long_side
            "--rgb",
            "--wb"
        ]

        progress_msg = await message.answer(f"⏳ {cfg['label']}: 0%")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
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

            # делаем JPEG-превью, чтобы не упасть по лимитам Telegram
            preview_path = None
            try:
                preview_path = make_preview_jpeg(output_png)
                photo_file = FSInputFile(preview_path)
                await message.answer_photo(photo_file, caption=cfg["label"], reply_markup=kb_instruction(uid, idx))
            except Exception:
                # резерв: шлём как документ (без превью)
                try:
                    doc = FSInputFile(output_png)
                    await message.answer_document(doc, caption=cfg["label"], reply_markup=kb_instruction(uid, idx))
                except Exception as e:
                    await message.answer(
                        f"❌ Ошибка при отправке результата «{cfg['label']}»:\n<code>{e}</code>",
                        parse_mode="HTML"
                    )

            results[str(idx)] = {"png": output_png, "xlsx": output_xlsx}

            # чистим превью
            if preview_path and os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                except:
                    pass

            # проверим, создалась ли инструкция
            if not os.path.exists(output_xlsx):
                await message.answer(
                    "⚠️ Инструкция Excel не найдена для этого варианта. "
                    "Убедитесь, что `generate.py` сохраняет файл вида `<output>_instruction.xlsx` в RGB-режиме."
                )
        else:
            all_ok = False
            try:
                await progress_msg.edit_text(f"❌ {cfg['label']}: ошибка генерации")
            except Exception:
                pass
            await message.answer(
                f"❌ Ошибка при генерации «{cfg['label']}»:\n<code>{stderr}</code>",
                parse_mode="HTML"
            )
            if os.path.exists(output_png):
                try:
                    os.remove(output_png)
                except:
                    pass

    user_results[uid] = results

    # снимаем 1 использование кода (за один цикл из 3 вариантов)
    dec_use(uid)

    # финальное дружелюбное сообщение
    if all_ok:
        await message.answer(
            "🎉 Все три варианта готовы! Выберите понравившийся и нажмите «📊 Получить инструкцию».\n\n"
            "Хотите попробовать другое фото? Нажмите «Загрузить ещё фото» ниже ⬇️",
            reply_markup=kb_more_status(uid)
        )
    else:
        await message.answer(
            "Готово с предупреждениями. Вы можете отправить другое фото, либо запросить инструкцию для удачно сгенерированных вариантов.",
            reply_markup=kb_more_status(uid)
        )

    # чистим входной файл
    if os.path.exists(input_path):
        try:
            os.remove(input_path)
        except:
            pass

# ================== ОБРАБОТКА КНОПОК ==================
@dp.callback_query(F.data.startswith("choose_"))
async def handle_choice(callback: CallbackQuery):
    _, uid, idx = callback.data.split("_")
    uid = int(uid)

    if uid not in user_results or idx not in user_results[uid]:
        await callback.answer("Результат не найден", show_alert=True)
        return

    files = user_results[uid][idx]
    xlsx_path = files["xlsx"]
    png_path = files["png"]

    if os.path.exists(xlsx_path):
        doc = FSInputFile(xlsx_path)
        await callback.message.answer_document(doc, caption="📊 Ваша инструкция (Excel)")
        try:
            os.remove(xlsx_path)
        except:
            pass
    else:
        await callback.message.answer("❌ Инструкция не найдена. Возможно, генерация прервалась.")

    if os.path.exists(png_path):
        try:
            os.remove(png_path)
        except:
            pass

    # убираем запись для выбранного варианта
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
        await callback.message.answer("ℹ️ Код не привязан. Отправьте `/status ВАШ_КОД`", parse_mode="Markdown")
        await callback.answer()
        return
    st = get_code_status(code)
    if not st:
        await callback.message.answer("❌ Код не найден.")
    else:
        await callback.message.answer(
            f"🔐 Код: *{code}*\n"
            f"✅ Доступно: *{st['left']}* из *{st['limit']}*",
            parse_mode="Markdown"
        )
    await callback.answer()

# ================== ЗАПУСК ==================
async def main():
    # сбрасываем webhook, чтобы не было конфликта с polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())