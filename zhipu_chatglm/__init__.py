import time
from datetime import datetime, timedelta, timezone
from threading import Lock

import httpx
import jwt
from nonebot import on_message, on_command
from nonebot import require
from nonebot.adapters.onebot.v11 import PrivateMessageEvent, GroupMessageEvent, MessageEvent
from nonebot.internal.params import Depends
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from nonebot.rule import to_me
from nonebot_plugin_alconna import on_alconna, Args, Alconna, Image, Match, UniMsg

from zhenxun.configs.utils import PluginExtraData
from zhenxun.utils.message import MessageUtils
from .ctrl_chat_files import *

require("nonebot_plugin_session")
require("nonebot_plugin_localstore")
require("nonebot_plugin_saa")

# TODO 添加线程锁(时间换空间)或创建多个任务(空间换时间)

"""
依赖:
aiofiles
httpx
PyJWT
"""

# 初始化
api_key = config.glm_api_key  # api密钥
max_token = config.glm_max_tokens  # 最大token
private = config.glm_private  # 是否启用私聊
BASE_model = config.glm_model  # 默认对话模型
BASE_picture_model = config.pic_vid_model  # 默认图片和视频识别模型
lock = Lock()  # 线程锁

__plugin_meta__ = PluginMetadata(
    name="ChatGLM",
    description="与ChatGLM聊天吧",
    usage=f"""
    指令：
        @机器人 [文本]: 与Ai对话
        清除聊天记录: 清除当前会话
        chat !img [文本?][图片本体]: 识别图片
        chat !sessions: 显示当前会话ID
        导入预设 [prompt]: 导入选择的预设
        列出预设: 列出所有预设
        列出模型: 列出所有模型
        切换模型 [model]: 切换模型 
        显示当前模型: 显示当前聊天使用的模型
    powered by -Mr.吴-
    """.strip(),
    config=Config,
    extra=PluginExtraData(
        author="Laowu",
        version="0.01",
    ).model_dump(),
)

ALL_MODELS_ENCODE = {
    'language_models': (
        'glm-4-plus', 'glm-4-0520', 'glm-4', 'glm-4-air', 'glm-4-airx', 'glm-4-long', 'glm-4-flashx', 'glm-4-flash',
        'glm-4v-plus', 'glm-4v', 'glm-4v-flash'),
    'text_to_images_models': ('cogview-3-plus', 'cogview-3'),
    'text_to_videos_models': ('cogvideox',),
    'agent_models': ('glm-4-alltools',),
    'code_models': ('codegeex-4',),
    'role_playing_models': ('charglm-4', 'emohaa'),
    'web_search_models': ('web-search-pro',)
}
storage_models: dict[int, tuple[str, str]] = {}  # 存储每个会话单独模型
BASE_URL = 'https://open.bigmodel.cn/api/paas/v4'
API_ENDPOINTS = {
    'language_models': f'{BASE_URL}/chat/completions',
    'text_to_images_models': f'{BASE_URL}/images/generations',
    'text_to_videos_models': f'{BASE_URL}/videos/generations',
    'agent_models': f'{BASE_URL}/chat/completions',
    'code_models': f'{BASE_URL}/chat/completions',
    'role_playing_models': f'{BASE_URL}/chat/completions',
    'web_search_models': f'{BASE_URL}/tools'
}


# 替换字符串中特殊字符
def replace_special_characters(text: str) -> str:
    escape_map = {
        "\n": "\\n",
        "\t": "\\t",
        "'": "\\'",
        '"': '\\"'
    }

    def replace(match):
        return escape_map[match.group(0)]

    pattern = re.compile(r'[\n\t\'"]')
    return pattern.sub(replace, text)


# 获取模型URL
def get_model_url(model: str) -> str | None:
    """
    输入模型编码，返回模型url
    :param model: 模型编码
    :return: 模型对应url或None
    """
    for model_type, models in ALL_MODELS_ENCODE.items():
        if model in models:
            return API_ENDPOINTS[model_type]
    return None


if not api_key:
    logger.error("没有配置api_key，插件将无法进行对话")


async def get_session_id(event: MessageEvent) -> int | None:
    """
    获取会话ID，群组/私聊
    :param event: 事件
    :return: 会话ID
    """
    if isinstance(event, GroupMessageEvent):
        logger.success(f"获取群聊ID", "ChatGLM", result=f"{event.group_id}")
        return event.group_id
    elif isinstance(event, PrivateMessageEvent) and private:
        logger.success(f"获取私聊ID", "ChatGLM", result=f"{event.user_id}")
        return event.user_id
    return None


get_session_id = Depends(get_session_id)


def is_lock_acquired():
    """
    检查锁是否被占用。
    :return: 如果锁未被占用，返回True；如果锁被占用，返回False。
    """
    acquired = lock.acquire(blocking=False)
    if acquired:
        lock.release()
    return acquired


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


# 请求模型对应API
# TODO 适配其他模型返回值
async def request_model(content, model_type, base_url, additional_params: dict = None) -> str:
    """
    请求API
    :param content: 请求内容
    :param model_type: 模型类型
    :param base_url: 模型对应API地址
    :param additional_params: 额外参数
    :return: 消息内容
    """
    with lock:
        if not api_key:
            raise ValueError("没有配置api_key，插件无法进行对话")
        auth_token = await generate_jwt(api_key)
        headers = {
            "Authorization": f"Bearer {auth_token}"
        }
        data = {
            "model": model_type,
            "temperature": config.glm_temperature,
            "messages": content
        }
        if additional_params:
            data.update(additional_params)
        try:
            logger.info(f"正在请求[{model_type}]")
            async with httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10, read=config.glm_timeout, write=20, pool=30)) as client:
                res = await client.post(base_url, headers=headers, json=data)
                res.raise_for_status()
                res = res.json()
        except httpx.ConnectTimeout as e:
            logger.error(f"ChatGLM请求超时: {e}")
            await MessageUtils.build_message(f'连接超时，请重试').finish()
        except httpx.HTTPError as e:
            logger.error(f"ChatGLM请求接口出错: {e}")
            await MessageUtils.build_message(f'请求接口出错').finish()
        logger.success("请求模型API", "ChatGLM", result=f"请求成功, 使用token: [{res['usage']['total_tokens']}]")
        try:
            res_raw = res['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            res_raw = res
        return str(res_raw)


_talk = on_message(
    priority=997,
    rule=to_me(),
    block=True
)

# 清除当前会话历史聊天记录
_clear_history = on_alconna(
    "清除聊天记录",
    priority=857,
    block=True
)

_identify_picture = on_alconna(
    Alconna("chat !img", Args["text?", str]["image?", Image]),
    priority=857,
    block=True
)

# 显示当前是否存在会话，存在则返回ID
_list_sessions = on_command(
    "chat !session",
    priority=857,
    block=True,
    permission=SUPERUSER
)

_import_prompt = on_alconna(
    Alconna("导入预设", Args["nickname", str]),
    priority=857,
    block=True
)

_list_prompt = on_command(
    '列出预设',
    priority=857,
    block=True
)

_list_all_model = on_command(
    '列出模型',
    priority=857,
    block=True
)

_change_model = on_alconna(
    Alconna('切换模型', Args["model?", str]),
    priority=857,
    block=True
)

_list_model = on_command(
    '显示当前模型',
    priority=857,
    block=True
)


@_talk.handle()
async def _(
        text: UniMsg,
        key_id: int | None = get_session_id,
):
    if not key_id:
        await _talk.finish()
    if not api_key:
        await _talk.finish("没有配置api_key，无法对话")
    if not text:
        await _talk.finish()
    if is_lock_acquired():
        log_file_path = log_dir / f"{key_id}.json"
        text = text.extract_plain_text()
        text = replace_special_characters(text)
        if (len(text) < 6 and random.random() < 0.7) or b_lexicon:
            if result := get_lexicon_result(text):
                await _talk.finish(result)
            elif b_lexicon:
                await _talk.finish("爪巴爪巴")
        else:
            await user_in(key_id, text)
            try:
                chat_history = await read_chat_history(log_file_path)
                if key_id not in storage_models:
                    storage_models[key_id] = (BASE_model, get_model_url(BASE_model))
                result = await request_model(chat_history, *storage_models[key_id])
                await ai_out(key_id, result)
                await MessageUtils.build_message(result).finish(reply_to=True)
            except json.JSONDecodeError as e:
                os.remove(log_file_path)
                logger.error(f"JSON转换错误: {e}")
                await _talk.finish(f'聊天记录炸了，已重置\n会话ID: {key_id}')
    else:
        await _talk.finish("当前锁正在被占有，请稍后再试")


@_clear_history.handle()
async def _(
        key_id: int | None = get_session_id,
):
    if not key_id:
        await _clear_history.finish()
    if is_lock_acquired():
        log_file_path = log_dir / f"{key_id}.json"
        if log_file_path.exists():
            os.remove(log_file_path)
            await _clear_history.finish("已清除当前会话")
        await _clear_history.finish("当前没有会话")
    else:
        await _clear_history.finish("当前锁正在被占有，请稍后再试")


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
        key_id: int | None = get_session_id
):
    if not key_id:
        await _identify_picture.finish()
    if not api_key:
        await _identify_picture.finish("没有配置api_key，无法对话")
    if not image:
        await _identify_picture.finish("没有输入图片 已结束")
    if is_lock_acquired():
        if (key_id not in storage_models) or (
                storage_models[key_id][0] not in {'glm-4v-plus-0111', 'glm-4v-plus', 'glm-4v', 'glm-4v-flash'}):
            storage_models[key_id] = (BASE_picture_model, get_model_url(BASE_picture_model))
        await MessageUtils.build_message("正在识别图片").send(reply_to=True)
        log_file_path = log_dir / f"{key_id}.json"
        text = replace_special_characters(text)
        await user_in(key_id, text, img=True, url=image.url)
        try:
            chat_history = await read_chat_history(log_file_path)
            result = await request_model(chat_history, *storage_models[key_id])
            await ai_out(key_id, result)
            await MessageUtils.build_message(result).finish(reply_to=True)
        except json.JSONDecodeError as e:
            os.remove(log_file_path)
            logger.error(f"JSON转换错误: {e}")
            await _identify_picture.finish(f'聊天记录炸了，已重置\n会话ID: {key_id}')
        except BaseException as e:
            logger.error(f"其他错误: {e}")
    else:
        await _identify_picture.finish("当前锁正在被占有，请稍后再试")


@_list_sessions.handle()
async def _(key_id: int | None = get_session_id):
    if not key_id:
        await _list_sessions.finish()
    log_file_path = log_dir / f"{key_id}.json"
    if log_file_path.exists():
        await _list_sessions.finish(f"当前会话存在\n会话ID: {key_id}")
    await _list_sessions.finish("当前没有会话")


@_import_prompt.handle()
async def _(nickname: str, key_id: int | None = get_session_id):
    if not key_id:
        await _import_prompt.finish()
    if not nicknames or (nickname not in nicknames):
        await _import_prompt.finish(f"没有{nickname}预设")
    if is_lock_acquired():
        result = await file_init(key_id, nickname)
        await _import_prompt.send(f"正在导入预设")
        message = f"{result}\n会话ID: {key_id}"
        await MessageUtils.build_message(message).finish(reply_to=True)
    else:
        await _import_prompt.finish("当前锁正在被占有，请稍后再试")


@_list_prompt.handle()
async def _(): await _list_prompt.finish(f"当前可导入的预设: {nicknames}")


@_change_model.handle()
async def _(model: str, key_id: int | None = get_session_id):
    if not key_id:
        await _change_model.finish()
    if not model:
        await _change_model.finish('没有输入模型')
    if key_id in storage_models.keys() and model == storage_models[key_id][0]:
        await _change_model.finish(f"当前模型: {model}")
    if is_lock_acquired():
        if url := get_model_url(model):
            storage_models[key_id] = (model, url)
            log_file_path = log_dir / f"{key_id}.json"
            if log_file_path.exists():
                os.remove(log_file_path)
            await _change_model.send("已清除历史聊天记录")
            logger.info(f"用户/群组切换模型[{model}]")
            await _change_model.finish(f'已切换模型{model}')
        await _change_model.finish(f'没有{model}模型')
    else:
        await _change_model.finish("当前锁正在被占有，请稍后再试")


@_list_model.handle()
async def _(key_id: int | None = get_session_id):
    if not key_id:
        await _list_model.finish()
    global storage_models
    if not storage_models or (key_id not in storage_models):
        await _list_model.finish(f'当前模型{BASE_model}')
    await _list_model.finish(f'当前模型{storage_models[key_id][0]}')


@_list_all_model.handle()
async def _(): await _list_all_model.finish(f"当前可用的模型: {ALL_MODELS_ENCODE}")
