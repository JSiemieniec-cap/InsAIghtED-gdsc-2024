from langchain_core.tools import tool
from sqlalchemy import text
from src.static.util import ENGINE
from typing import Literal


@tool
def get_answers_to_question(
        questionnaire_answers_table: Literal['StudentQuestionnaireAnswers', 'CurriculumQuestionnaireAnswers', 'HomeQuestionnaireAnswers', 'TeacherQuestionnaireAnswers', 'SchoolQuestionnaireAnswers'],
        question_code: str
) -> str:
    """
    When you know the question code but you don't know what are the possible anwers in the survey, query the database to find out what answers are available
    Query the database and returns possible answer to a given question

    Args:
        questionnaire_answers_table (str): the table related to the `general_table` containing answers.
        question_code (str): the code of the question or multiple questions. In return one gets the full list of possible answers to this questions returned.

    Returns:
        str: The list of all possible answers to the question with the code given in `question_code`.
        
        
    Example1:
    get_answers_to_question(
        questionnaire_answers_table = 'StudentQuestionnaireAnswers',
        question_code = 'ASBG01')
    
    Example2:
    get_answers_to_question(
        questionnaire_answers_table = 'StudentQuestionnaireAnswers',
        question_code = 'ASBG01, AFCG32')
    
    """
    in_clause = ','.join([f"'{i.strip()}'" for i in question_code.split(',')])
    query = f"""
        SELECT DISTINCT ATab.Code, ATab.Answer
        FROM {questionnaire_answers_table} AS ATab
        WHERE ATab.Code in ({in_clause})
    """

    with ENGINE.connect() as connection:
#    with db_engine.connect() as connection:
        try:
            res = connection.execute(text(query))
        except Exception as e:
            return f'Wrong query, encountered exception {e}.'

    ret = ""
    for result in res:
        ret += ", ".join(map(str, result)) + "\n"

    return ret

@tool
def query_database(query: str) -> str:
    """Query the PIRLS postgres database and return the results as a string.

    Args:
        query (str): The SQL query to execute.

    Returns:
        str: The results of the query as a string, where each row is separated by a newline.

    Raises:
        Exception: If the query is invalid or encounters an exception during execution.
    """
    # lower_query = query.lower()
    # record_limiters = ['count', 'where', 'limit', 'distinct', 'having', 'group by']
    # if not any(word in lower_query for word in record_limiters):
    #     return 'WARNING! The query you are about to perform has no record limitations! In case of large tables and ' \
    #            'joins this will return an incomprehensible output.'

    with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
        except Exception as e:
            return f'Wrong query, encountered exception {e}.'

    max_result_len = 3_000
    ret = '\n'.join(", ".join(map(str, result)) for result in res)
    if len(ret) > max_result_len:
        ret = ret[:max_result_len] + '...\n(results too long. Output truncated.)'

    return f'Query: {query}\nResult: {ret}'