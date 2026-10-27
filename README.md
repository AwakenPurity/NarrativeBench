<div align="center">

# NarrativeBench: Benchmarking Multi-trait Automated Scoring and Feedback Efficacy of Large Language Models for Chinese Narrative Essays

<a href='https://github.com/yxwang19/AlphaContext'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='paper/AlphaContext.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>


</div>
## 📌 Overview

**NarrativeBench** is a Chinese K-12 essay dataset designed for **automated essay scoring (AES)** and **educational NLP research**.  
It contains **4,500 student essays** annotated with **multi-dimensional scores and feedback** (`essay_total_data_4500_260417.xlsx`).

- Language: Chinese (zh)
- Domain: Primary school narrative writing
- Size: 4,500 essays
- Prompts: 14 writing topics

---

## 📊 Dataset Structure

Each sample corresponds to one student essay with the following fields:

| Field                | Description                 |
| -------------------- | --------------------------- |
| `hw_title`           | Homework title              |
| `hw_requirement`     | Assignment requirements     |
| `stu_title`          | Student's essay title       |
| `class_id`           | Anonymized class identifier |
| `content`            | Essay content               |
| `total_score`        | Overall score               |
| `relevance_score`    | Relevance to prompt         |
| `relevance_comment`  | Feedback on relevance       |
| `content_score`      | Content quality             |
| `content_comment`    | Feedback on content         |
| `expression_score`   | Language and fluency        |
| `expression_comment` | Feedback on expression      |

---

## 🧠 Annotation Protocol

- Annotators: 6 professional Chinese language teachers  
- Each essay is scored by **2 independent teachers**
- Evaluation dimensions:
  - **Relevance** (topic alignment)
  - **Content** (depth & quality)
  - **Expression** (fluency & style)
- Each dimension includes:
  - Numeric score
  - Natural language feedback

---

## 🎯 Use Cases

- Automated Essay Scoring (AES)
- Cross-prompt generalization evaluation
- Educational AI systems
- Feedback generation
- Chinese NLP benchmarks

---

## ⚖️ Responsible AI Considerations

### Data Collection

- Collected from real classroom assignments via an educational platform
- Fully anonymized (no personal identifiers)

### Potential Bias

- Limited to grades 3–5 Chinese students
- Focused on narrative writing only
- May reflect annotator subjectivity

### Limitations

- Not suitable for:
  - Other languages
  - Higher education levels
  - Non-narrative genres (e.g., argumentative)

