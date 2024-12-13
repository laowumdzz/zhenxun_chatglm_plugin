import json
import os
import random
import time
from datetime import datetime, timedelta, timezone

import aiofiles
import httpx
import jwt
from nonebot import get_plugin_config, require
from nonebot.adapters.onebot.v11 import PrivateMessageEvent, GroupMessageEvent, MessageEvent
from nonebot.params import Depends
from nonebot.plugin import PluginMetadata
from nonebot_plugin_alconna import on_alconna, Args, Alconna, Image, Match

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.configs.utils import PluginExtraData
from zhenxun.services.log import logger
from zhenxun.utils.message import MessageUtils
from .config import Config
from .config import Path

require("nonebot_plugin_session")
require("nonebot_plugin_localstore")
require("nonebot_plugin_saa")

"""
刚学会python和nonebot2不久，如有不满，轻点喷
刚学会python和nonebot2不久，如有不满，轻点喷
刚学会python和nonebot2不久，如有不满，轻点喷
"""

# TODO 修改chat为可替换
__plugin_meta__ = PluginMetadata(
    name="ChatGLM",
    description="与ChatGLM聊天吧",
    usage="""
    指令：
        talk [文本]: 与Ai对话
        chat !reset: 清除当前会话
        chat !img [文本?][图片本体]: 识别图片
        chat !sessions: 显示当前会话ID
        导入预设 [prompt]: 导入选择的预设
    """.strip(),
    config=Config,
    extra=PluginExtraData(
        author="Laowu",
        version="0.01",
    ).dict(),
)

config = get_plugin_config(Config)

cmd = config.glm_cmd
api_key = config.glm_api_key
temp = config.glm_history_path
prompt_path = config.prompt
max_token = config.glm_max_tokens
base_url = config.glm_api_addr
private = config.glm_private

temp.mkdir(parents=True, exist_ok=True)
log_dir = temp

if not api_key:
    logger.error("没有配置api_key，插件将无法进行对话")

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


# TODO 采用system方法定义预设
# 导入预设, user模式
async def file_init(key_id, nickname) -> str:
    log_file_path = log_dir / f"{key_id}.json"
    prompts = prompt[nickname]
    logger.info(f'创建/打开文件: {log_file_path}, 并导入 {nickname}预设')
    for index, value in enumerate(prompts['prompt_sys']):
        sys_data = json.dumps({"role": "user", "content": value}, ensure_ascii=False)
        ait_data = json.dumps({"role": "assistant", "content": prompts["prompt_ait"][index]}, ensure_ascii=False)
        await write_file(log_file_path, sys_data)
        await write_file(log_file_path, ait_data)
    return f"成功导入{nickname}预设"


# 写入历史聊天记录文件
async def write_file(log_file_path, json_data) -> None:
    try:
        if log_file_path.exists():
            async with aiofiles.open(log_file_path, 'r+', encoding='utf-8') as file:
                await file.seek(0, os.SEEK_END)
                # 如果文件不为空，则在现有内容后添加逗号和换行符
                if (await file.tell()) > 0:
                    await file.write(',\n')
                await file.write(json_data)
        else:
            async with aiofiles.open(log_file_path, 'w', encoding='utf-8') as file:
                await file.write(json_data)
    except Exception as e:
        logger.error(f"文件写入错误，日志: {e}")


# 读取历史聊天记录
async def read_chat_history(log_file_path) -> None | dict:
    if log_file_path.exists():
        async with aiofiles.open(log_file_path, encoding='utf-8') as file:
            logger.success("成功读取历史聊天记录", "读取历史聊天记录")
            return json.loads(f"[{await file.read()}]")
    return None


# 获取会话ID，群组/私聊
async def get_session_id(event: MessageEvent) -> int | None:
    if isinstance(event, GroupMessageEvent):
        logger.success(f"读取群聊ID: {event.group_id}", "获取群聊ID")
        return event.group_id
    elif isinstance(event, PrivateMessageEvent) and private:
        logger.success(f"读取私聊ID: {event.user_id}", "获取私聊ID")
        return event.user_id
    return None


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
    json_data = json.dumps(data, ensure_ascii=False)
    logger.info(f"用户/群组{key_id}将图片及文本{json_data}写入文件{log_file_path}", "Write content")
    await write_file(log_file_path, json_data)


# 用户输入
async def user_in(key_id, text):
    log_file_path = log_dir / f"{key_id}.json"
    data = {"role": "user", "content": text}
    json_data = json.dumps(data, ensure_ascii=False)
    logger.info(f"用户/群组{key_id}将文本{json_data}写入文件{log_file_path}", "Write content")
    await write_file(log_file_path, json_data)


# Ai输出
async def ai_out(key_id, text):
    log_file_path = log_dir / f"{key_id}.json"
    data = {"role": "assistant", "content": text}
    json_data = json.dumps(data, ensure_ascii=False)
    logger.info(f"ChatGLM返回文本{json_data}写入文件{log_file_path}", "Write content")
    await write_file(log_file_path, json_data)


# 生成JosnWebToken
async def generate_jwt(apikey: str):
    try:
        key_id, secret = apikey.split(".")
    except ValueError as e:
        await MessageUtils.build_message(f"错误的apikey！{e}").finish()
    payload = {
        "api_key": key_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=1),
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


# 随机选择真寻anime.json里对应键的文本
async def get_anime(text: str) -> str | None:
    keys = anime_data.keys()
    for key in keys:
        if text.find(key) != -1:
            return random.choice(anime_data[key])


# 异步请求API
async def request(auth_token, content, model=config.glm_model) -> str | None:
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }
    data = {
        "model": model,
        "temperature": config.glm_temperature,
        "messages": content
    }
    logger.info(f"当前[ChatGLM]对话使用模型：{model}")
    if max_token:
        data["max_tokens"] = max_token

    try:
        logger.info("正在请求ChatGLM")
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10, read=config.glm_timeout, write=20, pool=30)) as client:
            res = await client.post(base_url, headers=headers, json=data)
            res = res.json()
    except httpx.ConnectTimeout as e:
        logger.error(f"ChatGLM请求超时: {e}")
        await MessageUtils.build_message(f'连接超时，请重试').finish()
    except httpx.HTTPError as e:
        logger.error(f"ChatGLM请求接口出错: {e}")
        await MessageUtils.build_message(f'请求接口出错').finish()
    # noinspection PyBroadException
    try:
        res_raw = res['choices'][0]['message']['content']
    except BaseException as e:
        logger.error(f"res_raw切片出错: {e}")
        res_raw = res
    return str(res_raw)


_talk = on_alconna(
    Alconna(rf"{cmd}", Args["text", str]),
    priority=856,
    block=True
)

_clear_history = on_alconna(
    r"chat !reset",
    priority=857,
    block=True
)

_identify_picture = on_alconna(
    Alconna(r"chat !img", Args["text?", str]["image?", Image]),
    priority=857,
    block=True
)

_list_sessions = on_alconna(
    r"chat !session",
    priority=857,
    block=True
)

# TODO 加入列出所有预设功能
_import_prompt = on_alconna(
    Alconna(r"导入预设", Args["nickname", str]),
    priority=857,
    block=True
)


@_talk.handle()
async def _(
        text: str,
        key_id: int | None = Depends(get_session_id),
):
    if not key_id:
        await _talk.finish()
    if not api_key:
        await _talk.finish("没有配置api_key，无法对话")
    if not text:
        await _talk.finish()
    log_file_path = log_dir / f"{key_id}.json"
    text = text.replace("\n", "\\n").replace('\t', '\\t').replace("'", "\\'").replace('"', '\\"')
    if len(text) < 6 and random.random() < 0.7:
        if result := await get_anime(text):
            await MessageUtils.build_message(result).finish()
    await user_in(key_id, text)
    try:
        chat_history = await read_chat_history(log_file_path)
        auth = await generate_jwt(api_key)
        result = await request(auth, chat_history)
        await ai_out(key_id, result)
        await MessageUtils.build_message(result).finish(reply_to=True)
    except json.JSONDecodeError as e:
        os.remove(log_file_path)
        logger.error(f"JSON转换错误: {e}")
        await _talk.finish(f'聊天记录炸了，已重置\n会话ID: {key_id}')


@_clear_history.handle()
async def _(
        key_id: int | None = Depends(get_session_id),
):
    if not key_id:
        await _clear_history.finish()
    log_file_path = log_dir / f"{key_id}.json"
    if log_file_path.exists():
        os.remove(log_file_path)
        await _clear_history.finish("已清除当前会话")
    await _clear_history.finish("当前没有会话")


@_identify_picture.handle()
async def _(text: Match[str], image: Match[Image]):
    if text.available:
        _identify_picture.set_path_arg("text", text.result)
    else:
        _identify_picture.set_path_arg("text", "识别这张图片的内容")
    if image.available:
        _identify_picture.set_path_arg("image", image.result)


@_identify_picture.got_path("image", prompt="请输入图片")
async def _(
        text: str,
        image: Image,
        key_id: int | None = Depends(get_session_id)
):
    if not key_id:
        await _identify_picture.finish()
    if not api_key:
        await _identify_picture.finish("没有配置api_key，无法对话")
    if not image:
        await _identify_picture.finish("没有输入图片 已结束")
    await MessageUtils.build_message("正在识别图片").send(reply_to=True)
    log_file_path = log_dir / f"{key_id}.json"
    text = text.replace("\n", "\\n").replace('\t', '\\t').replace("'", "\\'").replace('"', '\\"')
    await user_img(key_id, image.url, text)
    try:
        chat_history = await read_chat_history(log_file_path)
        auth = await generate_jwt(api_key)
        result = await request(auth, chat_history, config.pic_vid_model)
        await ai_out(key_id, result)
        await MessageUtils.build_message(result).finish(reply_to=True)
    except json.JSONDecodeError as e:
        os.remove(log_file_path)
        logger.error(f"JSON转换错误: {e}")
        await _identify_picture.finish(f'聊天记录炸了，已重置\n会话ID: {key_id}')


@_list_sessions.handle()
async def _(key_id: int | None = Depends(get_session_id)):
    if not key_id:
        await _list_sessions.finish("当前没有会话")
    log_file_path = log_dir / f"{key_id}.json"
    await _list_sessions.send("正在检测当前会话")
    if log_file_path.exists():
        await _list_sessions.finish(f"当前会话存在\n会话ID: {key_id}")
    await _list_sessions.finish("当前没有会话")


@_import_prompt.handle()
async def _(nickname: str, key_id: int | None = Depends(get_session_id)):
    if not key_id:
        await _import_prompt.finish()
    if not nicknames or (nickname not in nicknames):
        await _import_prompt.finish(f"没有{nickname}预设")
    result = await file_init(key_id, nickname)
    await _import_prompt.send(f"正在导入预设")
    message = f"{result}\n会话ID: {key_id}"
    await MessageUtils.build_message(message).finish(reply_to=True)
