from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    # TODO 删除使用命令形式，群聊采用@触发，私聊直接触发
    glm_cmd: str = "talk"
    """插件命令"""

    glm_api_key: str = ""
    """你的智谱apikey"""

    glm_history_path: str | Path = Path() / "resources" / "chat_history"
    """聊天记录储存路径"""

    prompt: str | Path = Path() / "data" / "prompt.json"
    """预设文件路径"""

    glm_max_tokens: int = 0
    """最大输出的token,0为不限"""

    # TODO 加入更多模型
    glm_model: str = "glm-4-plus"
    """目前该插件支持的模型有：glm-4, glm-4-plus, glm-4v, glm-4v-plus"""

    pic_vid_model: str = "glm-4v"
    """图片及视频识别模型: glm-4v, glm-4v-plus"""

    glm_temperature: float = 0.6
    """采样温度，控制输出的随机性，必须为正数，取值范围是0.0～1.0，不能等于 0，值越大，会使输出更随机，更具创造性；值越小，输出会更加稳定或确定"""

    glm_timeout: int = 60
    """响应超时时间，默认为60秒"""

    # TODO  添加绘画和视频生成功能
    # glm_draw: bool = True
    # """是否启用ai画图及生成视频功能"""

    glm_private: bool = True
    """是否启用私聊 仅Onebot V11"""

    # TODO 群聊通过@触发而无需使用命令
    # glm_at: bool = True
    # """是否通过@触发聊天 仅Onebot V11"""
