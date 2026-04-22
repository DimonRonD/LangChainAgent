import os
import json
import logging

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agent")

try:
    from langchain.agents import create_agent as _create_agent

    def build_agent(llm, tools):
        return _create_agent(llm, tools=tools)

except Exception:
    from langgraph.prebuilt import create_react_agent as _create_react_agent

    def build_agent(llm, tools):
        return _create_react_agent(llm, tools=tools)


# 1) Локальная модель через Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    base_url=OLLAMA_BASE_URL,
)


# 2) Инструмент, который агент может вызывать
@tool
def multiply(a: int, b: int) -> int:
    """Умножает два целых числа a и b."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Делит a на b."""
    if b == 0:
        raise ValueError("Деление на ноль невозможно.")
    return a / b

# 3) Собираем агента
TOOLS = [multiply, divide]
TOOL_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}
agent = build_agent(llm, tools=TOOLS)

def _run_tool_call_from_text(text: str):
    """Пытается выполнить tool-call, если модель вернула его обычным JSON-текстом."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    name = payload.get("name")
    parameters = payload.get("parameters", {})
    tool_ = TOOL_BY_NAME.get(name)
    if not tool_:
        return None

    if not isinstance(parameters, dict):
        raise ValueError("Некорректные параметры инструмента.")

    logger.info("Fallback tool-call detected: name=%s parameters=%s", name, parameters)
    tool_result = tool_.invoke(parameters)
    logger.info("Fallback tool-call result: %s", tool_result)
    return tool_result

def _serialize_message_content(content):
    """Нормализует контент сообщений в строку для логирования."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)

def ask_agent(text: str) -> str:
    """Отправляет одно сообщение агенту и возвращает финальный ответ."""
    payload = {"messages": [HumanMessage(content=text)]}
    logger.info(
        "Model request: model=%s temperature=%s base_url=%s user_text=%s",
        OLLAMA_MODEL,
        OLLAMA_TEMPERATURE,
        OLLAMA_BASE_URL,
        text,
    )

    result = agent.invoke(payload)
    logger.info("Raw agent result: %s", result)

    final_text = result["messages"][-1].content
    final_text = _serialize_message_content(final_text)
    logger.info("Model raw response content: %s", final_text)

    # Некоторые модели через Ollama иногда возвращают tool-call как обычный JSON.
    tool_result = _run_tool_call_from_text(final_text)
    if tool_result is not None:
        final_answer = str(tool_result)
        logger.info("Final response after fallback tool execution: %s", final_answer)
        return final_answer

    logger.info("Final response without fallback: %s", final_text)
    return final_text

# 4) Интерактивный режим
print("Локальный агент запущен ✅")
print("Примеры: 'Сколько будет 7 умножить на 8?' или '22 разделить на 11'")
print("Выход: exit / quit\n")

while True:
    user_text = input("Ты: ").strip()
    if user_text.lower() in ("exit", "quit"):
        print("Пока 👋")
        break

    try:
        answer = ask_agent(user_text)
        print("Агент:", answer, "\n")
    except Exception as e:
        print("Ошибка:", e)
        print("Проверь, что Ollama запущена и модель скачана: ollama list\n")
