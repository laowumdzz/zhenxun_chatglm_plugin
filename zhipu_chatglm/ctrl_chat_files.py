import json
import os

import aiofiles
from nonebot import get_plugin_config

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.log import logger
from .config import Config
from .config import Path

config = get_plugin_config(Config)
prompt_path = config.prompt

temp = config.glm_history_path
temp.mkdir(parents=True, exist_ok=True)
log_dir = temp

# 获取zhenxun_anime内容
with open(DATA_PATH / "anime.json", encoding="utf8") as anime_file:
    anime_data = json.load(anime_file)


# 获取预设文件所有预设
def get_prompt() -> tuple | None:
    with open(prompt_path, encoding="utf-8") as prompt_file:
        prompt_file.seek(0, os.SEEK_END)
        file_size = prompt_file.tell()
        logger.info(f"预设文件大小: {file_size}")
        if file_size != 0:
            prompt_file.seek(0)
            logger.success("读取预设文件", "get_prompt")
            prompts: dict = json.load(prompt_file)
            return prompts, prompts.keys()
        return None, None


# 判断预设文件是否存在并读取
if ((isinstance(prompt_path, Path) and prompt_path.exists()) or
        (isinstance(prompt_path, str) and os.path.exists(prompt_path))):
    prompt, nicknames = get_prompt()
    nicknames = list(nicknames) if nicknames else None


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
    logger.info(f'创建/打开文件: {log_file_path}, 并导入 {nickname}预设')
    for index, value in enumerate(prompts['prompt_sys']):
        sys_data = {"role": "system", "content": value}
        ait_data = {"role": "assistant", "content": prompts["prompt_ait"][index]}
        await write_file(log_file_path, sys_data)
        await write_file(log_file_path, ait_data)
    return f"成功导入{nickname}预设"


# 读取历史聊天记录
async def read_chat_history(log_file_path) -> None | list:
    try:
        if log_file_path.exists():
            async with aiofiles.open(log_file_path, encoding='utf-8') as file:
                logger.info("读取历史聊天记录", "读取历史聊天记录")
                content = await file.read()
                return json.loads(content) if content else None
        return None
    except Exception as e:
        logger.error(f"文件读取错误，日志: {e}")


# 写入图片识别文本
async def user_img(key_id, url, text):
    log_file_path = log_dir / f"{key_id}.json"
    data = {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url}}
        ]
    }
    logger.info(f"[用户/群组]{key_id}将图片及文本{data}写入文件{log_file_path}", "Write content")
    await write_file(log_file_path, data)


# 用户输入
async def user_in(key_id, text):
    log_file_path = log_dir / f"{key_id}.json"
    data = {"role": "user", "content": text}
    logger.info(f"[用户/群组] [{key_id}] 将文本 [{data}] 写入文件 [{log_file_path}]", "Write content")
    await write_file(log_file_path, data)


# Ai输出
async def ai_out(key_id, text):
    log_file_path = log_dir / f"{key_id}.json"
    data = {"role": "assistant", "content": text}
    logger.info(f"[ChatGLM]返回文本 [{data}] 写入文件 [{log_file_path}]", "Write content")
    await write_file(log_file_path, data)
