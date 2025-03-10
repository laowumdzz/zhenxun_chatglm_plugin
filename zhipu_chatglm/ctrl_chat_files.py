import json
import os

import aiofiles
from nonebot import get_driver
from nonebot import get_plugin_config

from zhenxun.services.log import logger
from .config import Config, Path

config = get_plugin_config(Config)

default_prompt_path = config.prompt
log_dir = config.glm_history_path
log_dir.mkdir(parents=True, exist_ok=True)

driver = get_driver()


async def clear_history():
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    logger.info('清除历史聊天记录', 'chatglm_plugin')


driver.on_startup(clear_history)


# 获取预设文件所有预设
def get_prompt(prompt_path: Path | str) -> tuple | None:
    prompt_path = Path(prompt_path) if isinstance(prompt_path, str) else prompt_path

    if not prompt_path.exists():
        return None, None

    with prompt_path.open(encoding="utf-8") as prompt_file:
        prompt_file.seek(0, os.SEEK_END)
        file_size = prompt_file.tell()
        logger.info(f"预设文件大小: {file_size}")
        if file_size == 0:
            return None, None
        prompt_file.seek(0)
        logger.success("读取预设文件", "ChatGLM", {"文件路径": prompt_path}, "True")
        prompts: dict = json.load(prompt_file)
        return prompts, list(prompts.keys())


# 判断预设文件是否存在并读取
prompt, nicknames = get_prompt(default_prompt_path)


# 写入历史聊天记录文件
async def write_file(log_file_path: Path, dict_data: dict) -> None:
    try:
        if isinstance(dict_data, dict):
            if log_file_path.exists():
                async with aiofiles.open(log_file_path, 'r+', encoding='utf-8') as file:
                    content = await file.read()
                    data = json.loads(content) if content else []
                    data.append(dict_data)
                    await file.seek(0)
                    await file.truncate()
                    await file.write(json.dumps(data, ensure_ascii=False, indent=4))
            else:
                async with aiofiles.open(log_file_path, 'w', encoding='utf-8') as file:
                    await file.write(json.dumps([dict_data], ensure_ascii=False, indent=4))
    except Exception as e:
        logger.error(f"文件写入错误，日志: {e}")


# 导入预设，system模式
# 注意：GLM-4V及衍生不支持system模式
async def file_init(key_id, nickname) -> str:
    log_file_path = log_dir / f"{key_id}.json"
    prompts = prompt[nickname]
    for index, value in enumerate(prompts['prompt_sys']):
        sys_data = {"role": "system", "content": value}
        ait_data = {"role": "assistant", "content": prompts["prompt_ait"][index]}
        await write_file(log_file_path, sys_data)
        await write_file(log_file_path, ait_data)
    logger.success("导入预设", "ChatGLM", {"会话ID": key_id, "预设": nickname}, "True")
    return f"成功导入{nickname}预设"


# 读取历史聊天记录
async def read_chat_history(log_file_path) -> None | list:
    try:
        if log_file_path.exists():
            async with aiofiles.open(log_file_path, encoding='utf-8') as file:
                logger.success("读取历史聊天记录", "ChatGLM", {"文件路径": log_file_path}, "True")
                content = await file.read()
                return json.loads(content) if content else None
        return None
    except Exception as e:
        logger.error(f"文件读取错误，日志: {e}")


# 用户输入
async def user_in(key_id, text, img=False, url=None):
    log_file_path = log_dir / f"{key_id}.json"
    if img:
        data = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        }
    else:
        data = {"role": "user", "content": text}
    logger.info(f"[用户/群组] [{key_id}] 将文本或图片链接 [{data}] 写入文件 [{log_file_path}]", "ChatGLM")
    await write_file(log_file_path, data)


# Ai输出
async def ai_out(key_id, text):
    log_file_path = log_dir / f"{key_id}.json"
    data = {"role": "assistant", "content": text}
    logger.info(f"[ChatGLM]返回文本 [{data}] 写入文件 [{log_file_path}]", "ChatGLM")
    await write_file(log_file_path, data)
