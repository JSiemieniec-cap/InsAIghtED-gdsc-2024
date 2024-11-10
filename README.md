# Team InsAIghtED solution repo :blue_book:

![Banner](http://gdsc-bucket-058264313357.s3.amazonaws.com/insighted_banner_team.png)

## Table of contents:
1. [Solution Concept](#solution-concept)
2. [Details of the solution](#details-of-the-solution)
3. [How to run model](#how-to-run-model)
4. [Repo structure](#repo-structure)

## Solution Concept
Our solution is a combination of multi-agent system post-processed by LLM chains with addition of human preprocessed inputs.

Also we enforce the well structured output of an answer:
1. **Short answer** is the most important part. It should answer the question in one sentence. For all those users who are not interested in reading the whole "essays" this should suffice.
2. **Data Visualization** which presents the data in appropriate chart, sometimes adding the context to the result. 
3. **Details** section is a complex and more detailed answer to the given question. It provides explanation how model came to such conclusion providing data and insights. It also shows more information related to topic and shows it in good looking style.
4. **Fun Section** - although it might be considered as not serious, we strongly believe it's a crucial part of the solution which goal is to highlight the necessity to make fourth-grader enjoy their reading activities. :smiley:

## Details of the solution
Our solution consists of 
1. **Human-preprocessed data and insights** - to process and extract essential information from PIRLS database
2. **RAG (Retrieval-Augmented Generation)** - to add external sources to make answer more extented
3. **Multi-Agent solution** - using Crew AI, it is the core of the solution.
    There are 4 agents:
    - Lead Data Analyst
    - Data Engineer
    - Chart Preparer
    - Data Scientist
4. **Postprocessing chains** - to capture all information and represents it in scheduled structure

## How to run model

installing the necessary libraries
```
!pip install -r ./requirements.txt
```
Load environmental modules
```
import dotenv

#Utils
from src.static.ChatBedrockWrapper import ChatBedrockWrapper
from src.static.submission import Submission
# main function
from src.submission.crews.advanced_PIRLS_crew_rag_gdp import AdvancedPIRLSCrew

dotenv.load_dotenv()
```
Set up Large language model and Crew of Agents
```
llm = ChatBedrockWrapper(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={'temperature': 0, 'max_tokens': 200000},
        call_id=12321
    )
crew = AdvancedPIRLSCrew(llm=llm)
```
Asking LLM model
```
%%time
query = "Which country had a reading score closest to 547 for fourth-grade students in the PIRLS 2021 study?"
answer = crew.run(query)
```
Showing result
```
# allows displaying the proper formats
from IPython.display import display, Markdown
display(Markdown(answer))
```

Note: Running those examples requires setting up credentials to Amazon Bedrock and S3 bucket used by the team

All above code you can reach in Usage_Examples.ipynb and execute cells to test a model.

## Repo structure

```
├── rag
├── src
    ├── static
    └── submission
        ├── config
            ├── agents_rag_gdp.yaml
            └── tasks_rag_gdp.yaml
        ├── crew
            └── advanced_PIRLS_crew_rag_gdp.py
        ├── tools
            ├── __init__.py
            ├── database.py
            └── research_tools.py
        └── create_submission.py
├── tests
    └── tests.ipynb
├── readme.md
├── requirements.txt
└── Usage_Examples.ipynb
```
#### `rag`
- **Description**: Folder containing files implementing algorithms using Retrieval-Augmented Generation (RAG). It is used to assist the model in the process of generating answers to questions, relying on external data sources.

#### `src/submission/config/agents_rag_gdp.yaml`
- **Description**: YAML file configuring how agents in CrewAI should behave

#### `src/submission/config/tasks_rag_gdp.yaml`
- **Description**: YAML file configuring how agents in CrewAI should perform the tasks assigned to them

#### `src/submission/crew/advanced_PIRLS_crew_rag_gdp.py`
- **Description**: The main file containing the implementation of the `AdvancedPIRLSCrew` class. This class is responsible for managing agents, the RAG system and coordinating their activities. It uses settings from YAML files to assign appropriate tasks to agents that process data and generate results. It also provides fun section.

#### `src/submission/tools/database.py`
- **Description**: A file containing the methods the model uses to answer the question asked

#### `src/submission/tools/research_tools.py`
- **Description**: A file that allows you to externally browse the Internet to find more accurate data to get the best possible answer

#### `requirements.txt`
- **Description**: A file with a list of dependencies (libraries) required to run the project. Allows you to quickly install all necessary packages using the `pip` utility.

#### `Usage_Examples.ipynb`
- **Description**: Jupyter Notebook containing examples of how to use the model. It documents how to run queries, present results, and test the model's performance. It may also contain additional comments and instructions related to the use of the system.

## How to Test Main Functionality
--- 
In the 'tests' folder, you can find a notebook containing all the tests, which will verify the main functions of the model and data types.
