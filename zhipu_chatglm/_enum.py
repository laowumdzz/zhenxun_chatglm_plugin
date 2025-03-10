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
