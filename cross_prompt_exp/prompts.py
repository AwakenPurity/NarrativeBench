# -*- coding: utf-8 -*-
"""
Prompt templates for cross-prompt LLM essay scoring experiments.
"""

# ============================================================
# System prompt — defines the scoring rubric
# ============================================================
SYSTEM_PROMPT = """你是一个经验丰富的语文作文评分专家。请对以下学生作文进行评分。

## 评分维度说明

### 1. Relevance（审题，0-8分）
- 8分：文章完全掌握了题目要求。解读透彻，文章始终紧扣主题。
- 7分：文章对题目把握得当，紧扣主题，展现了出色的题目分析能力。
- 6分：文章对题目掌握良好，但部分内容与主题不够契合。
- 5分：文章对题目把握较好，但未能始终保持主题的连贯性。
- 4分：文章对题目有基本理解，但经常出现偏离主题的情况。
- 3分：文章抓住了基本要求，但未能充分体现主题。
- 2分：文章未能正确理解题目，明显偏离主题。
- 1分：文章在题干分析方面存在明显错误，缺乏基本题目理解能力。
- 0分：文章与题目完全无关，对主题的理解完全错误。

### 2. Content（内容，0-8分）
- 8分：内容丰富，词汇量大，展现了出色的词汇量和措辞句法技巧。
- 7分：内容充实，用词精准，体现了独创性和写作技巧。
- 6分：内容丰富，运用了有效词汇和句式，但词汇多样性有待提升。
- 5分：内容尚可，能吸引读者，但深度和广度有提升空间。
- 4分：部分传达了意图，但表达效果较弱。
- 3分：展现了基本表达能力，但内容不够丰富，词汇和句式需润色。
- 2分：表达枯燥，缺乏必要的词汇积累和表达技巧。
- 1分：基本表达能力不足，内容缺乏深度。
- 0分：内容过于简单，严重缺乏细节。

### 3. Expression（表达，0-8分）
- 8分：文章流畅，逻辑连贯，无明显语法错误。
- 7分：逻辑流畅，表达准确，错误极少，结构清晰。
- 6分：表达相对流畅，逻辑清晰，但存在少量语法错误。
- 5分：逻辑相对清晰准确，但细节和精准度需完善。
- 4分：基本流畅，但存在明显语法错误。
- 3分：逻辑基本连贯，但有语法错误，不影响基本含义。
- 2分：表达不连贯，逻辑中断频繁，大量语法错误。
- 1分：严重逻辑漏洞，频繁语法错误阻碍含义传达。
- 0分：表达极度不连贯，过多语法和拼写错误。

### 4. Total Score（总分，0-8分）
- 8分：主题契合度、内容和表达均出色，体现了扎实的写作能力。
- 7分：整体质量优秀，至少两个方面表现突出。
- 6分：质量良好，大部分方面表现稳定，存在小问题。
- 5分：质量中等，满足基本要求，但存在明显不足。
- 4分：完成度尚可，至少一个方面存在明显问题。
- 3分：质量较差，存在多方面缺陷。
- 2分：各方面均存在不足，可读性差。
- 1分：整体质量差，理解不清、内容不完整、表达混乱。
- 0分：严重偏离任务，内容无关，结构混乱。"""

# ============================================================
# Few-shot example template
# ============================================================
FEW_SHOT_EXAMPLE_TEMPLATE = """## 示例作文
- 作文标题：{stu_title}
- 作文内容：{content}

## 示例评分
{{
 "total_score": {total_score},
 "relevance_score": {relevance_score},
 "content_score": {content_score},
 "expression_score": {expression_score}
}}"""

# ============================================================
# User prompt template with few-shot examples
# ============================================================
USER_PROMPT_TEMPLATE = """## 作业信息
- 作业标题：{hw_title}
- 作业要求：{hw_requirement}

## 学生作文
- 作文标题：{stu_title}
- 作文内容：{content}

## 输出要求
请严格按照以下 JSON Schema 格式输出评分结果，不要输出其他内容：
{{
 "total_score": 总分(0-8的整数),
 "relevance_score": 相关性得分(0-8的整数),
 "content_score": 内容得分(0-8的整数),
 "expression_score": 表达得分(0-8的整数)
}}"""


def build_few_shot_prompt(examples):
    """Build few-shot examples string.

    Args:
        examples: list of dicts with keys stu_title, content, total_score,
                  relevance_score, content_score, expression_score

    Returns:
        str: formatted few-shot examples
    """
    parts = []
    for ex in examples:
        parts.append(FEW_SHOT_EXAMPLE_TEMPLATE.format(
            stu_title=ex["stu_title"],
            content=ex["content"],
            total_score=ex["total_score"],
            relevance_score=ex["relevance_score"],
            content_score=ex["content_score"],
            expression_score=ex["expression_score"],
        ))
    return "\n\n".join(parts)


def build_messages(hw_title, hw_requirement, stu_title, content, few_shot_examples=None):
    """Build OpenAI-compatible messages for scoring.

    Args:
        hw_title: assignment title
        hw_requirement: assignment requirements
        stu_title: student essay title
        content: student essay content
        few_shot_examples: optional list of example dicts

    Returns:
        list: messages for OpenAI API
    """
    system_content = SYSTEM_PROMPT

    if few_shot_examples:
        few_shot_str = build_few_shot_prompt(few_shot_examples)
        user_content = (
            f"以下是几个评分示例，请参考这些示例的评分标准进行评分。\n\n"
            f"{few_shot_str}\n\n---\n\n"
        ) + USER_PROMPT_TEMPLATE.format(
            hw_title=hw_title,
            hw_requirement=hw_requirement,
            stu_title=stu_title,
            content=content,
        )
    else:
        user_content = USER_PROMPT_TEMPLATE.format(
            hw_title=hw_title,
            hw_requirement=hw_requirement,
            stu_title=stu_title,
            content=content,
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
