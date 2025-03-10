import json
import random
import re
from pathlib import Path
from typing import Any

from nonebot_plugin_alconna import At, AtAll, UniMessage
from tortoise import Tortoise, fields
from typing_extensions import Self

from zhenxun.configs.config import BotConfig
from zhenxun.services.db_context import Model
from zhenxun.utils.message import MessageUtils
from ._enum import WordType
from .ctrl_chat_files import config

lexicon = config.lexicon_files
lexicon = Path(lexicon) if isinstance(lexicon, str) else lexicon
lexicon = (lexicon,) if not isinstance(lexicon, tuple) else lexicon


# 词库: 词条数据库操作
class WordBank(Model):
    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    """自增id"""
    word_type = fields.IntField(default=WordType.FUZZY.value)
    """匹配规则"""
    problem = fields.TextField()
    """问题"""
    answer = fields.TextField()
    """回答"""

    class Meta:
        table = "word_bank"
        table_description = "词条数据库"

    # noinspection PyMethodOverriding
    @classmethod
    async def exists(cls,
                     problem: str,
                     answer: str | None,
                     word_type: WordType | None = None,
                     ) -> bool:
        """
        检测问题是否存在

        :param problem: 问题
        :param answer: 回答
        :param word_type: 匹配规则
        :return: True -> 存在, False -> 不存在
        """
        query = cls.filter(problem=problem)
        if answer:
            query = query.filter(answer=answer)
        if word_type is not None:
            query = query.filter(word_type=word_type.value)
        return await query.exists()

    @classmethod
    async def check_problem(
            cls,
            problem: str,
            word_type: WordType,
    ) -> list | None:
        """
        检测是否包含该问题并获取所有回答
        :param problem: 问题
        :param word_type: 匹配规则
        :return: 回答列表
        """
        db = Tortoise.get_connection("default")
        db_class_name = BotConfig.get_sql_type()
        if word_type == WordType.EXACT.value:
            # 完全匹配
            if data_list := await cls.filter(word_type__not=WordType.REGEX.value, problem=problem).all():
                return data_list
        elif word_type == WordType.FUZZY.value:
            # 模糊匹配
            """
            match db_class_name:
                case "postgres":
                    sql = (cls.filter(word_type=WordType.REGEX.value).sql() + " AND POSITION(problem IN $1) > 0")
                case "sqlite":
                    sql = (cls.filter(word_type=WordType.REGEX.value).sql() + " AND INSTR(?, problem) > 0")
                case "mysql":
                    sql = (cls.filter(word_type=WordType.REGEX.value).sql() + " AND INSTR(%s, problem) > 0")
                case _:
                    raise Exception(f"Unsupported database type: {db_class_name}")
            """

            # data_list = await cls.filter(word_type=WordType.FUZZY.value, problem__istartswith=problem).all()

            if data_list := await cls.filter(problem__istartswith=problem).all():
                return [data for data in data_list if len(data.problem) <= len(problem) + 1]

            # if data_list := await db.execute_query_dict(sql, [problem]):
            #     return [cls(**data) for data in data_list]
        elif word_type == WordType.REGEX.value:
            # 正则匹配
            match db_class_name:
                case "postgres":
                    sql = (cls.filter(word_type=WordType.REGEX.value).sql() + " AND $1 ~ problem")
                case "sqlite":
                    sql = (cls.filter(word_type=WordType.REGEX.value).sql() + " AND problem LIKE ?")
                    problem = f"%{problem}%"
                case "mysql":
                    sql = (cls.filter(word_type=WordType.REGEX.value).sql() + " AND problem REGEXP ?")
                case _:
                    raise Exception(f"Unsupported database type: {db_class_name}")

            if data_list := await db.execute_query_dict(sql, [problem]):
                return [cls(**data) for data in data_list]
        return None

    @classmethod
    async def _format2answer(
            cls,
            problem: str,
            answer: str,
            query: Self | None = None,
    ) -> UniMessage:
        """
        将占位符转换为实际内容
        :param problem: 问题内容
        :param answer: 回答内容
        :param query: 查询语句
        :return: 消息内容
        """
        if not query:
            query = await cls.get_or_none(
                problem=problem,
                answer=answer,
            )
        if not answer:
            answer = str(query.answer)  # type: ignore
        if query and query.placeholder:
            type_list = re.findall(r"\[(.*?):placeholder_.*?]", answer)
            answer_split = re.split(r"\[.*?:placeholder_.*?]", answer)
            placeholder_split = query.placeholder.split(",")
            result_list = []
            for index, ans in enumerate(answer_split):
                result_list.append(ans)
                if index < len(type_list):
                    t = type_list[index]
                    p = placeholder_split[index]
                    if t == "at":
                        if p == "0":
                            result_list.append(AtAll())
                        else:
                            result_list.append(At(flag="user", target=p))
            return MessageUtils.build_message(result_list)
        return MessageUtils.build_message(answer)

    @classmethod
    async def get_answer(
            cls,
            problem: str,
            word_type: WordType = WordType.FUZZY.value,
    ) -> tuple[str, str] | None:
        """
        根据问题内容获取随机回答
        :param problem: 问题内容
        :param word_type: 匹配规则
        :return: 消息内容
        """
        if data_list := await cls.check_problem(problem, word_type):
            random_answer = random.choice(data_list)
            # if random_answer.word_type == WordType.REGEX.value:
            #     r = re.search(random_answer.problem, problem)
            #     has_placeholder = re.search(r"\$(\d)", random_answer.answer)
            #     if r and r.groups() and has_placeholder:
            #         pats = re.sub(r"\$(\d)", r"\\\1", random_answer.answer)
            #         random_answer.answer = re.sub(random_answer.problem, pats, problem)
            # return (
            #     await cls._format2answer(
            #         random_answer.problem,
            #         random_answer.answer,
            #     )
            #     if random_answer.placeholder
            #     else MessageUtils.build_message(random_answer.answer)
            # )
            return random_answer.answer, random_answer.problem

    @classmethod
    async def get_problem_all_answer(
            cls,
            problem: str,
            index: int | None = None,
    ) -> tuple[str, list[UniMessage]]:
        """
        获取指定问题所有回答
        :param problem: 问题
        :param index: 下标
        :return: 问题和所有回答
        """
        if index is not None:
            _problem = (await cls.filter().order_by("id").values_list("problem", flat=True))
            sort_problem = []
            for p in _problem:
                if p not in sort_problem:
                    sort_problem.append(p)
            if index > len(sort_problem) - 1:
                return "下标错误，必须小于问题数量...", []
            problem = sort_problem[index]
        f = cls.filter(problem=problem)
        answer_list = await f.all()
        if not answer_list:
            return "词条不存在...", []
        return problem, [await cls._format2answer("", "", a) for a in answer_list]

    @classmethod
    def _int2type(cls, value: int) -> str:
        for key, member in WordType.__members__.items():
            if member.value == value:
                return key
        return ""

    @classmethod
    def _handle_problem(cls, problem_list: list["WordBank"]) -> list[tuple[Any, str]]:
        """
        格式化处理问题
        :param problem_list: 消息列表
        :return: 格式化后的消息列表
        """
        _tmp = []
        result_list = []
        for q in problem_list:
            if q.problem not in _tmp:
                result_list.append((q.problem, "-",))
                _tmp.append(q.problem)
        return result_list


class ImportHelper:
    @classmethod
    def to_create_list(cls,
                       problem: str,
                       answer_list: list[str],
                       word_type: WordType = WordType.FUZZY.value
                       ) -> list[WordBank]:
        """
        获取创建列表
        :param problem: 问题
        :param answer_list: 回答列表
        :param word_type: 匹配规则
        :return: 问答列表
        """
        create_list = []
        for answer in answer_list:
            create_list.append(
                WordBank(
                    word_type=word_type,
                    problem=problem,
                    answer=answer,
                )
            )
        return create_list

    @classmethod
    async def import_word(cls) -> str:
        """
        导入词条
        异常: FileNotFoundError: 文件不存在
        """
        data = dict()
        for file in lexicon:
            try:
                if file.exists():
                    with file.open(encoding="utf-8") as file_content:
                        data.update(json.load(file_content))
                else:
                    raise FileNotFoundError(f"文件不存在: {file}")
            except json.JSONDecodeError:
                raise json.JSONDecodeError(f"从文件解码JSON时出错: {file}", "", 0)
            except Exception as ee:
                raise Exception(f"处理文件时出错[{file}]: {ee}")
        create_list = []
        for problem, answer_list in data.items():
            create_list += cls.to_create_list(problem, answer_list)
        await WordBank.bulk_create(create_list, 100)
        return f"成功导入 {len(create_list)} 条词条！"
