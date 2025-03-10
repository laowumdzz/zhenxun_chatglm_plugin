from enum import StrEnum, Enum


class ChatType(StrEnum):
    """
    监听对象
    """
    PRIVATE = "PRIVATE"
    """私聊"""
    GROUP = "GROUP"
    """群聊"""
    ALL = "ALL"
    """全局"""


# 模型: 模型编码
class ModelsEncoe(Enum):
    language_models = (
        'glm-4-plus', 'glm-4-0520', 'glm-4', 'glm-4-air', 'glm-4-airx', 'glm-4-long', 'glm-4-flashx', 'glm-4-flash',
        'glm-4v-plus', 'glm-4v', 'glm-4v-flash')
    text_to_images_models = ('cogview-3-plus', 'cogview-3')
    text_to_videos_models = ('cogvideox',)
    agent_models = ('glm-4-alltools',)
    code_models = ('codegeex-4',)
    role_playing_models = ('charglm-4', 'emohaa')
    web_search_models = ('web-search-pro',)


# 模型: 模型链接
class ModelsApisLink(StrEnum):
    BASE_URL = 'https://open.bigmodel.cn/api/paas/v4'
    language_models = f'{BASE_URL}/chat/completions'
    text_to_images_models = f'{BASE_URL}/images/generations'
    text_to_videos_models = f'{BASE_URL}/videos/generations'
    agent_models = f'{BASE_URL}/chat/completions'
    code_models = f'{BASE_URL}/chat/completions'
    role_playing_models = f'{BASE_URL}/chat/completions'
    web_search_models = f'{BASE_URL}/tools'


# 词库: 关键词匹配规则
class WordType(Enum):
    text = None
    EXACT = 0
    """精准"""
    FUZZY = 1
    """模糊"""
    REGEX = 2
    """正则"""


WordType.EXACT.text = "精准"
WordType.FUZZY.text = "模糊"
WordType.REGEX.text = "正则"
