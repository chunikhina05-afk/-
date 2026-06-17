import os
import json
import psycopg2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pyvis.network import Network
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

load_dotenv()
TOKEN = os.environ["TELEGRAM_TOKEN"]
DATABASE_URL = os.environ["DATABASE_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)

TYPE_LABELS = {
    "Person": "👤 Люди", "Company": "🏢 Компании", "Project": "📁 Проекты",
    "Task": "✅ Задачи", "Idea": "💡 Идеи", "Technology": "⚙️ Технологии",
    "Book": "📚 Книги", "Event": "📅 События", "Link": "🔗 Ссылки",
}
TYPE_COLORS = {
    "Person": "#4f8cff", "Company": "#ff9f40", "Project": "#9b59b6", "Task": "#2ecc71",
    "Idea": "#f1c40f", "Technology": "#1abc9c", "Book": "#e74c3c", "Event": "#e67e22", "Link": "#95a5a6",
}


def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS notes (
            id SERIAL PRIMARY KEY, telegram_id BIGINT, text TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        ALTER TABLE notes ADD COLUMN IF NOT EXISTS embedding vector(768);
        ALTER TABLE notes ADD COLUMN IF NOT EXISTS tags TEXT;
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY, telegram_id BIGINT, type TEXT, name TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE (telegram_id, type, name)
        );
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS description TEXT;
        CREATE TABLE IF NOT EXISTS relations (
            id SERIAL PRIMARY KEY, telegram_id BIGINT,
            source_id INTEGER REFERENCES entities(id),
            target_id INTEGER REFERENCES entities(id),
            relation_type TEXT, created_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS entity_history (
            id SERIAL PRIMARY KEY, telegram_id BIGINT,
            entity_id INTEGER REFERENCES entities(id),
            change TEXT, created_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def embed(text):
    result = client.models.embed_content(
        model="gemini-embedding-001", contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return result.embeddings[0].values


def to_vec(values):
    return "[" + ",".join(str(x) for x in values) + "]"


def extract(text):
    prompt = (
        "Разбери сообщение пользователя для личной базы знаний. Верни ТОЛЬКО валидный JSON, без markdown:\n"
        '{"entities": [{"type": "...", "name": "...", "description": "..."}], '
        '"relations": [{"source": "...", "target": "...", "type": "..."}], '
        '"tags": ["...", "..."], "ask": "..."}\n\n'
        "Типы entities: Person, Company, Project, Task, Idea, Technology, Book, Event, Link.\n"
        "Типы relations: WORKS_AT, KNOWS, PARTICIPATED_IN, RELATED_TO, CREATED, TASK_FOR, REFERENCES.\n\n"
        "ПРАВИЛА:\n"
        "- 'запиши что...', 'заметка:', 'идея:' — это Idea. НЕ делай из мысли отдельную Task и не выдумывай связи.\n"
        "- Конкретное дело с действием — это Task. Имя короткое, с глаголом.\n"
        "- name короткое (1–4 слова). Факты про сущность клади в description (кратко). Нет фактов — description = \"\".\n"
        "- relations создавай ТОЛЬКО если связь явно следует из текста.\n"
        "- tags: 1–4 коротких тега-категории в нижнем регистре. Непонятно — [].\n\n"
        "ПОЛЕ ask: если это задача/напоминание и не хватает детали (кому, когда) — задай ОДИН короткий вопрос. "
        'Иначе ask = "".\n\n'
        f"Сообщение: {text}"
    )
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception:
        return {"entities": [], "relations": [], "tags": [], "ask": ""}
    data.setdefault("entities", [])
    data.setdefault("relations", [])
    data.setdefault("tags", [])
    data.setdefault("ask", "")
    return data


def find_entity_in_text(user_id, text):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT name FROM entities WHERE telegram_id = %s", (user_id,))
    names = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close()
    tl = text.lower()
    matches = [n for n in names if n and n.lower() in tl]
    return max(matches, key=len) if matches else None


def build_graph_text(user_id, name):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT id, name, description FROM entities WHERE telegram_id = %s AND lower(name) LIKE %s LIMIT 1",
                (user_id, f"%{name.lower()}%"))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return f"Не нашёл «{name}» в базе."
    eid, ename, desc = row
    cur.execute("""
        SELECT r.relation_type, e2.name, 'out' FROM relations r JOIN entities e2 ON r.target_id = e2.id WHERE r.source_id = %s
        UNION ALL
        SELECT r.relation_type, e1.name, 'in' FROM relations r JOIN entities e1 ON r.source_id = e1.id WHERE r.target_id = %s
    """, (eid, eid))
    rows = cur.fetchall()
    cur.close(); conn.close()
    header = ename + (f" — {desc}" if desc else "")
    if not rows:
        return f"{header}\n └── (пока нет связей)"
    lines = [header]
    for rt, other, direction in rows:
        lines.append(f" ├── {rt} → {other}" if direction == "out" else f" ├── {other} → {rt}")
    return "\n".join(lines)


def build_history_text(user_id, name):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM entities WHERE telegram_id = %s AND lower(name) LIKE %s LIMIT 1",
                (user_id, f"%{name.lower()}%"))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return f"Не нашёл «{name}» в базе."
    eid, ename = row
    cur.execute("SELECT change, created_at FROM entity_history WHERE entity_id = %s ORDER BY created_at", (eid,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    if not rows:
        return f"📜 По «{ename}» пока нет истории."
    lines = [f"📜 История «{ename}»:"]
    for change, created in rows:
        lines.append(f"• {created.strftime('%d.%m.%Y %H:%M')}: {change}")
    return "\n".join(lines)


def build_graph_html(user_id):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT id, type, name FROM entities WHERE telegram_id = %s", (user_id,))
    ents = cur.fetchall()
    cur.execute("SELECT source_id, target_id, relation_type FROM relations WHERE telegram_id = %s", (user_id,))
    rels = cur.fetchall()
    cur.close(); conn.close()
    if not ents:
        return None
    net = Network(height="800px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    # Разносим узлы, чтобы не слипались и подписи не налезали
    net.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=200,
                   spring_strength=0.05, damping=0.09, overlap=1)
    ids = set()
    for eid, etype, name in ents:
        net.add_node(eid, label=name, title=etype,
                     color=TYPE_COLORS.get(etype, "#888888"),
                     size=24, font={"size": 18})
        ids.add(eid)
    for src, tgt, rtype in rels:
        if src in ids and tgt in ids:
            net.add_edge(src, tgt, label=rtype, font={"size": 12, "color": "#777777"})
    path = "knowledge_graph.html"
    net.save_graph(path)
    return path
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT id, type, name FROM entities WHERE telegram_id = %s", (user_id,))
    ents = cur.fetchall()
    cur.execute("SELECT source_id, target_id, relation_type FROM relations WHERE telegram_id = %s", (user_id,))
    rels = cur.fetchall()
    cur.close(); conn.close()
    if not ents:
        return None
    net = Network(height="750px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    ids = set()
    for eid, etype, name in ents:
        net.add_node(eid, label=name, title=etype, color=TYPE_COLORS.get(etype, "#888888"))
        ids.add(eid)
    for src, tgt, rtype in rels:
        if src in ids and tgt in ids:
            net.add_edge(src, tgt, label=rtype)
    path = "knowledge_graph.html"
    net.save_graph(path)
    return path


def answer_question(user_id, question, last_entity=None):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    try:
        qvec = to_vec(embed(question))
        cur.execute("""
            SELECT text FROM notes WHERE telegram_id = %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector LIMIT 5
        """, (user_id, qvec))
        notes = [row[0] for row in cur.fetchall()]
    except Exception as err:
        print("Поиск по смыслу не сработал:", err)
        conn.rollback()
        cur.execute("SELECT text FROM notes WHERE telegram_id = %s ORDER BY created_at DESC LIMIT 10", (user_id,))
        notes = [row[0] for row in cur.fetchall()]

    focus = find_entity_in_text(user_id, question) or last_entity
    focus_block = ""
    if focus:
        cur.execute("SELECT id, name, description FROM entities WHERE telegram_id = %s AND lower(name) LIKE %s LIMIT 1",
                    (user_id, f"%{focus.lower()}%"))
        frow = cur.fetchone()
        if frow:
            fid, fname, fdesc = frow
            cur.execute("""
                SELECT r.relation_type, e2.name FROM relations r JOIN entities e2 ON r.target_id = e2.id WHERE r.source_id = %s
                UNION ALL
                SELECT r.relation_type, e1.name FROM relations r JOIN entities e1 ON r.source_id = e1.id WHERE r.target_id = %s
            """, (fid, fid))
            neigh = cur.fetchall()
            fl = [f"ФОКУС ВОПРОСА: {fname}" + (f" — {fdesc}" if fdesc else "")]
            related = {fname}
            for rt, other in neigh:
                fl.append(f"  {fname} {rt} {other}")
                related.add(other)
            focus_block = "\n".join(fl)
            for nm in related:
                cur.execute("SELECT text FROM notes WHERE telegram_id = %s AND text ILIKE %s LIMIT 3",
                            (user_id, f"%{nm}%"))
                for (t,) in cur.fetchall():
                    if t not in notes:
                        notes.append(t)

    cur.execute("SELECT type, name, description FROM entities WHERE telegram_id = %s", (user_id,))
    entities = cur.fetchall()
    cur.execute("""
        SELECT e1.name, r.relation_type, e2.name FROM relations r
        JOIN entities e1 ON r.source_id = e1.id JOIN entities e2 ON r.target_id = e2.id
        WHERE r.telegram_id = %s
    """, (user_id,))
    rels = cur.fetchall()
    cur.close(); conn.close()

    parts = []
    if focus_block:
        parts.append(focus_block + "\n")
    parts.append("СУЩНОСТИ:")
    for t, n, d in entities:
        parts.append(f"- [{t}] {n}" + (f" — {d}" if d else ""))
    parts.append("\nСВЯЗИ:")
    for s, rt, tg in rels:
        parts.append(f"- {s} {rt} {tg}")
    parts.append("\nЗАМЕТКИ:")
    for n in notes:
        parts.append(f"- {n}")
    context_text = "\n".join(parts)

    hint = f"Если вопрос неполный или с местоимениями — речь, скорее всего, о: {last_entity}.\n" if last_entity else ""
    prompt = (
        "Ты — помощник по личной базе знаний. " + hint +
        "Ответь, опираясь ТОЛЬКО на данные ниже. Если ответа нет — честно скажи. Отвечай по-русски, кратко.\n\n"
        f"=== ДАННЫЕ ===\n{context_text}\n\n=== ВОПРОС ===\n{question}"
    )
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    user_id = update.message.from_user.id
    low = text.lower()

    if low.startswith("/визуал") or low.startswith("/картинка") or low.startswith("/picture"):
        path = build_graph_html(user_id)
        if not path:
            await update.message.reply_text("В базе пока нет сущностей для графа.")
        else:
            await update.message.reply_document(document=open(path, "rb"), filename="knowledge_graph.html")
        return

    if low.startswith("/graph"):
        name = text[6:].strip()
        await update.message.reply_text("🕸️ " + build_graph_text(user_id, name) if name else "Напиши имя: /graph Иван")
        return

    if low.startswith("/история") or low.startswith("/history"):
        p = text.split(maxsplit=1)
        name = p[1].strip() if len(p) > 1 else ""
        await update.message.reply_text(build_history_text(user_id, name) if name else "Напиши имя: /история Иван")
        return

    if low.startswith("/тег") or low.startswith("/tag"):
        p = text.split(maxsplit=1)
        tag = p[1].strip().lower() if len(p) > 1 else ""
        if not tag:
            await update.message.reply_text("Напиши тег: например  /тег ai")
            return
        conn = psycopg2.connect(DATABASE_URL); cur = conn.cursor()
        cur.execute("SELECT text FROM notes WHERE telegram_id = %s AND tags ILIKE %s ORDER BY created_at DESC LIMIT 20",
                    (user_id, f"%{tag}%"))
        rows = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        if not rows:
            await update.message.reply_text(f"По тегу «{tag}» заметок не нашёл.")
        else:
            await update.message.reply_text((f"🏷️ Заметки по тегу «{tag}»:\n" + "\n".join(f"• {r}" for r in rows))[:4000])
        return

    if text.startswith("?") or text.endswith("?"):
        question = text.strip("?").strip()
        last_entity = context.user_data.get("last_entity")
        try:
            ans = answer_question(user_id, question, last_entity)
            await update.message.reply_text(f"🔎 {ans}")
            found = find_entity_in_text(user_id, question)
            if found:
                context.user_data["last_entity"] = found
        except Exception as err:
            print("Ошибка при ответе:", err)
            await update.message.reply_text("⚠️ Не получилось обработать вопрос (возможно, лимит Gemini). Попробуй через минуту.")
        return

    combined = False
    if context.user_data.get("pending_original"):
        text = context.user_data.pop("pending_original") + ". " + text
        combined = True

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("INSERT INTO notes (telegram_id, text) VALUES (%s, %s) RETURNING id", (user_id, text))
    note_id = cur.fetchone()[0]
    conn.commit()
    try:
        cur.execute("UPDATE notes SET embedding = %s::vector WHERE id = %s", (to_vec(embed(text)), note_id))
        conn.commit()
    except Exception as err:
        conn.rollback()
        print("Эмбеддинг не сохранён:", err)

    try:
        data = extract(text)
    except Exception as err:
        print("Ошибка разбора:", err)
        cur.close(); conn.close()
        await update.message.reply_text("💾 Записал, но разобрать пока не вышло (возможно, лимит Gemini).")
        return

    if data.get("ask") and not combined:
        cur.close(); conn.close()
        context.user_data["pending_original"] = text
        await update.message.reply_text(f"💾 Записал.\n\n🤔 Уточни, пожалуйста: {data['ask']}")
        return

    tags = ",".join(t.strip().lower() for t in data.get("tags", []) if t and t.strip())
    cur.execute("UPDATE notes SET tags = %s WHERE id = %s", (tags, note_id))

    name_to_id = {}
    for ent in data["entities"]:
        etype = ent.get("type")
        name = (ent.get("name") or "").strip()
        desc = (ent.get("description") or "").strip()
        if not name or etype not in TYPE_LABELS:
            continue
        cur.execute(
            """
            INSERT INTO entities (telegram_id, type, name, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (telegram_id, type, name)
            DO UPDATE SET description = COALESCE(NULLIF(EXCLUDED.description, ''), entities.description)
            RETURNING id
            """,
            (user_id, etype, name, desc),
        )
        eid = cur.fetchone()[0]
        name_to_id[name] = eid
        cur.execute("INSERT INTO entity_history (telegram_id, entity_id, change) VALUES (%s, %s, %s)",
                    (user_id, eid, text[:300]))

    for r in data["relations"]:
        src = name_to_id.get((r.get("source") or "").strip())
        tgt = name_to_id.get((r.get("target") or "").strip())
        rtype = r.get("type")
        if src and tgt and rtype:
            cur.execute(
                "INSERT INTO relations (telegram_id, source_id, target_id, relation_type) VALUES (%s, %s, %s, %s)",
                (user_id, src, tgt, rtype),
            )

    conn.commit()
    cur.close()
    conn.close()

    if not data["entities"] and not tags:
        await update.message.reply_text("💾 Записал.")
        return

    by_type = {}
    for ent in data["entities"]:
        etype = ent.get("type")
        name = (ent.get("name") or "").strip()
        desc = (ent.get("description") or "").strip()
        if name and etype in TYPE_LABELS:
            by_type.setdefault(etype, []).append(name + (f" ({desc})" if desc else ""))

    lines = ["💾 Сохранил и разложил по полочкам:\n"]
    for etype, names in by_type.items():
        lines.append(f"{TYPE_LABELS[etype]}: " + ", ".join(names))
    rel_lines = []
    for r in data["relations"]:
        s, t, rt = r.get("source"), r.get("target"), r.get("type")
        if s and t and rt:
            rel_lines.append(f"  • {s} → {rt} → {t}")
    if rel_lines:
        lines.append("\n🔗 Связи:")
        lines.extend(rel_lines)
    if tags:
        lines.append("\n🏷️ Теги: " + tags.replace(",", ", "))

    await update.message.reply_text("\n".join(lines))


print("Запускаю бота... чтобы остановить — нажми Ctrl+C")
init_db()
app = Application.builder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT, on_message))
app.run_polling()