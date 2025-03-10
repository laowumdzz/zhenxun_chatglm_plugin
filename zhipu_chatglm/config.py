from pathlib import Path

from pydantic import BaseModel

from zhenxun.configs.path_config import DATA_PATH, TEMP_PATH
from ._enum import ChatType, WordType


class Config(BaseModel):
    glm_api_key: str
    """你的智谱apikey"""

    glm_history_path: str | Path = TEMP_PATH / "chat_history"
    """聊天记录储存路径"""

    prompt: str | Path = DATA_PATH / "prompt.json"
    """预设文件路径"""

    lexicon_files: tuple[Path] | Path | str = (DATA_PATH / "anime.json", DATA_PATH / "cute_anime.json")
    """预先定义好的词库回复内容文件"""

    only_lexicon: bool = False
    """是否只使用词库回答"""

    glm_max_tokens: int = 0
    """最大输出的token,0为不限"""

    glm_model: str = "glm-4-flash"
    """默认模型，目前该插件支持的模型有：('glm-4-plus', 'glm-4-0520', 'glm-4', 'glm-4-air', 'glm-4-airx', 'glm-4-long', 'glm-4-flashx', 
    'glm-4-flash', 'glm-4v-plus', 'glm-4v', 'glm-4v-flash'), ('charglm-4', 'emohaa')"""

    pic_vid_model: str = "glm-4v"
    """默认模型，图片及视频识别模型: glm-4v, glm-4v-plus"""

    glm_temperature: float = 0.6
    """采样温度，控制输出的随机性，必须为正数，取值范围是0.0～1.0，不能等于 0，值越大，会使输出更随机，更具创造性；值越小，输出会更加稳定或确定"""

    glm_timeout: int = 60
    """响应超时时间，默认为60秒"""

    # TODO  添加绘画和视频生成功能
    glm_draw: bool = True
    """是否启用ai画图及生成视频功能"""

    glm_private: bool = True
    """是否启用私聊 仅Onebot V11"""

    listen_type: ChatType = ChatType.ALL
    """监听类型, 默认全局"""

    match_rule: WordType = WordType.FUZZY
    """匹配规则, 默认模糊匹配"""
