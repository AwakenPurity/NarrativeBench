# CROISSANT VALIDATION REPORT
================================================================================
## VALIDATION RESULTS
--------------------------------------------------------------------------------
Starting validation for file: croissant_NarrativeBench.json
### JSON Format Validation
✓
The file is valid JSON.
### Croissant Schema Validation
✓
The dataset passes Croissant validation.
### Responsible AI Metadata
✓
All required Responsible AI metadata fields are present.
### Records Generation Test
✓
No record sets found to validate.
## JSON-LD REFERENCE
================================================================================
```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "dct": "http://purl.org/dc/terms/",
    "sc": "https://schema.org/",
    "@base": "cr_base_iri/"
  },
  "@type": "sc:Dataset",
  "cr:conformsTo": "http://mlcommons.org/croissant/1.0",
  "name": "NarrativeBench",
  "description": "NarrativeBench is a dataset of 4,500 Chinese student essays with multi-dimensional scoring annotations. Each essay is associated with a homework title, requirement, student title, class ID, essay content, and scores with comments across three dimensions: relevance, content, and expression.",
  "url": "https://anonymous.4open.science/r/NarrativeBench",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "1.0.0",
  "inLanguage": "zh",
  "datePublished": "2025-04-26",
  "citation": "Zheng et al. (2025). NarrativeBench: A Chinese K-12 Essay Scoring Dataset from the Elion System.",
  "keywords": [
    "Chinese essay",
    "automated essay scoring",
    "natural language processing",
    "education",
    "narrative writing"
  ],
  "distribution": [
    {
      "@type": "sc:FileObject",
      "@id": "essay-data-file",
      "name": "essay_total_data_4500_260417.xlsx",
      "description": "Main dataset file containing 4,500 student essays with scoring annotations",
      "contentUrl": "https://anonymous.4open.science/r/NarrativeBench/essay_total_data_4500_260417.xlsx",
      "encodingFormat": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "sha256": "c2a2cf8cf282cd1ea6916c1d0a092d17f26397a63b32cb0a15dcb20e6b230a7a"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "@id": "essays",
      "name": "essays",
      "description": "Student essay records with multi-dimensional scoring",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "essays/hw_title",
          "name": "hw_title",
          "description": "The title of the homework assignment",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "hw_title"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/hw_requirement",
          "name": "hw_requirement",
          "description": "The detailed requirements of the homework assignment",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "hw_requirement"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/stu_title",
          "name": "stu_title",
          "description": "The title of the student's essay",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "stu_title"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/class_id",
          "name": "class_id",
          "description": "The class identifier of the student",
          "dataType": "sc:Integer",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "class_id"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/content",
          "name": "content",
          "description": "The full text content of the student's essay",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "content"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/total_score",
          "name": "total_score",
          "description": "The overall score of the essay",
          "dataType": "sc:Integer",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "total_score"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/relevance_score",
          "name": "relevance_score",
          "description": "Score for relevance to the homework topic and requirements",
          "dataType": "sc:Integer",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "relevance_score"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/relevance_comment",
          "name": "relevance_comment",
          "description": "Textual comment explaining the relevance score",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "relevance_comment"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/content_score",
          "name": "content_score",
          "description": "Score for the quality and depth of the essay content",
          "dataType": "sc:Integer",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "content_score"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/content_comment",
          "name": "content_comment",
          "description": "Textual comment explaining the content score",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "content_comment"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/expression_score",
          "name": "expression_score",
          "description": "Score for language expression, fluency, and writing style",
          "dataType": "sc:Integer",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "expression_score"
            }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "essays/expression_comment",
          "name": "expression_comment",
          "description": "Textual comment explaining the expression score",
          "dataType": "sc:Text",
          "source": {
            "fileObject": {
              "@id": "essay-data-file"
            },
            "extract": {
              "column": "expression_comment"
            }
          }
        }
      ]
    }
  ],
  "rai:dataCollection": "The dataset is derived from the Elion system, a professional essay collection platform tailored for Chinese K-12 education. Essays were collected from students as part of regular classroom writing assignments.",
  "rai:dataCollectionType": "Human-generated essays collected from an educational platform for Chinese K-12 students. All student identifying information has been anonymized prior to release.",
  "rai:dataUseCases": "Automated essay scoring, Chinese NLP, educational AI, writing quality assessment",
  "rai:dataBiases": "The dataset comprises 4,500 essays covering 14 writing prompts, authored by students in grades 3 to 5. The data reflects the curriculum and writing styles of Chinese primary school education and may not generalize to other grade levels, regions, or writing systems.",
  "rai:personalSensitiveInformation": "All student personal information has been anonymized. No names or other personally identifiable information are included in the released dataset. Only anonymized class IDs are retained.",
  "rai:annotationsPerItem": "Each essay was independently evaluated by two of six invited Chinese Language teachers across three dimensions: relevance, content, and expression. Each dimension includes a numeric score and a textual comment.",
  "rai:annotatorDemographics": "Six professional Chinese Language teachers were invited as annotators. Each essay was independently scored by two teachers to ensure annotation quality and consistency.",
  "rai:dataLimitations": "The dataset is limited to narrative writing essays from Chinese primary school students in grades 3 to 5, covering 14 specific writing prompts. It may not generalize to other writing genres (e.g., argumentative or expository), other education levels, or other languages. Scoring reflects the judgment of Chinese Language teachers and may carry annotator subjectivity.",
  "rai:dataSocialImpact": "NarrativeBench can facilitate the development of automated essay scoring systems for Chinese K-12 education, potentially reducing teacher workload and providing timely feedback to students. However, automated scoring systems trained on this data should be used as assistive tools rather than replacements for human judgment, to avoid reinforcing scoring biases or disadvantaging students whose writing styles differ from the training distribution.",
  "rai:hasSyntheticData": false
}
```