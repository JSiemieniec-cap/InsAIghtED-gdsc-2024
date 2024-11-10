from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from src.static.submission import Submission
from src.static.util import PROJECT_ROOT
import src.submission.tools.database as db_tools
import src.submission.tools.research_tools as research_tools


@CrewBase
class AdvancedPIRLSCrew(Submission):
    """Data Analysis Crew for the GDSC project.
    
    Main method is run:
    Input: prompt
    Output: answer to the prompt in markdown format
    
    Usage example: 
    crew = AdvancedPIRLSCrew()
    crew.run('How many students participated in PIRLS 2021?')
    
    """
    # Load the files from the config directory
    agents_config = PROJECT_ROOT / 'submission' / 'config' / 'agents_rag_gdp.yaml'
    tasks_config = PROJECT_ROOT / 'submission' / 'config' / 'tasks_rag_gdp.yaml'

    def __init__(self, llm):
        self.llm = llm

    def run(self, prompt: str) -> str:
        """
        run is the main method of AdvancedPIRLSCrew class.
        Input: prompt
        Output: answer to the prompt in markdown format

        Usage example: 
        crew = AdvancedPIRLSCrew()
        crew.run('How many students participated in PIRLS 2021')
        
        It starts with retrieval part from available sources, then it enhances the original prompt with additional retrieved knowledge and ask the crew to execute the whole process. Once crew returns the answer, it utilizes the chains implemented as additional methods of the class to extract short answer, complex answer (details), visualization and dad joke or meme to the fun section. Everything in order to structure the response properly.

        """
        import random
        # first section is rag - without any crew orchestration
        import boto3
        import os
        
        # download the data from S3
        files = ['7b08d22f-fe86-4bfe-a546-051e34289f4b/length.bin',
        '7b08d22f-fe86-4bfe-a546-051e34289f4b/link_lists.bin',
        '7b08d22f-fe86-4bfe-a546-051e34289f4b/data_level0.bin',
        '7b08d22f-fe86-4bfe-a546-051e34289f4b/header.bin',
        'chroma.sqlite3']

        # make directories locally
        main_folder = 'rag'
        directory_name = f'{main_folder}/collections_2'
        subfolder_name = '7b08d22f-fe86-4bfe-a546-051e34289f4b/'
        try:
            os.mkdir(f'./{main_folder}')
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            os.mkdir(f'./{directory_name}')
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            os.mkdir(f'./{directory_name}/{subfolder_name}')
        except Exception as e:
            print(f"An error occurred: {e}")

        s3 = boto3.resource('s3')
        for file in files:
            s3.meta.client.download_file('gdsc-bucket-058264313357', f'{directory_name}/{file}', f'./{directory_name}/{file}')
        
        # Set up RAG
        
        from chromadb.utils import embedding_functions
        import chromadb
        default_ef = embedding_functions.DefaultEmbeddingFunction()

        client = chromadb.PersistentClient(path=f"./{directory_name}")
        collection = client.get_collection(name="pirls_2021", embedding_function=default_ef)    
        
        # retrieval
        rag_result = collection.query(query_texts = [prompt], n_results=20)
        
        # prepare sources of external data
        sources = [source['source'].replace('https://www.youtube.com/watch?v=2D1RnQhyAZU', '["PIRLS 2021– Findings, IEA Education"](https://www.youtube.com/watch?v=2D1RnQhyAZU)').replace('https://www.youtube.com/watch?v=wACy8bzeOAU', '["What can we learn from PIRLS 2021?, Department of Education, University of Oxford"](https://www.youtube.com/watch?v=wACy8bzeOAU)') for source in rag_result['metadatas'][0]]
        documents = rag_result['documents'][0]
        sources_documents = ['source: '+l[0]+', content: '+l[1] for l in list(zip(sources, documents))]
        
        # enhance prompt
        rag_prompt = '.\n'.join(sources_documents).replace('PEARLS', 'PIRLS')
        new_prompt = f"""
        Answer this query {prompt}.
        Use the knowledge from this source in the final answer if it only helps:
        {rag_prompt}
        And add in the final answer all the sources (unique i.e. only once) of the relevant pieces of this knowledge if it was useful.
        """
        answer_all = self.crew().kickoff(inputs={'user_question': new_prompt})
        answer = answer_all.raw
        short_answer = self.short_answer(prompt, answer)
        complex_answer = self.complex_answer(prompt, answer)
        data_chart_answer = self.data_chart_answer(prompt, answer)
        chart_markdown = self.extract_markdown_data_scientist(answer_all.tasks_output[0].raw)
        if chart_markdown == "''":
            chart_markdown = ''
        chart_section = ""
        if data_chart_answer != "''":
            try:
                chart_section = f"""
> #### Data visualization
{chart_markdown}

!["Lack of appropriate data for visualization purpose"]({self.make_a_chart(data_chart_answer.replace("plot.show()","").replace("plt.show()",""))})
            """
            except:
                pass

        joke_section = ""
        
        if random.random() < 0.5:
            joke_section = f"""

*😂 You got a meme to smile for the rest of the day 😂*

![Meme](https://gdsc-bucket-058264313357.s3.amazonaws.com/img/insighted_meme_{random.randint(1,8)}.png)

            """
        else:
            joke_section = "*😂Do you want to see a joke related to the topic? WARNING!: it won't be funny 😂*"
            dad_joke_answer = self.dad_joke(prompt, answer)
            joke_section = joke_section + f"""

{dad_joke_answer}

            """
        
        final_answer = f"""
![Banner](https://gdsc-bucket-058264313357.s3.amazonaws.com/img/insighted_banner.jpg)

> #### Short answer

{short_answer}

{chart_section}


> #### Details

{complex_answer}


> #### *😂Fun section 😂*

{joke_section}

"""
        return final_answer
    
    def short_answer(self, query: str, answer: str) -> str:
        """
        Based on the original prompt/query and the answer generated by the crew, it creates a short and precise answer utilizing langchain chains. 
        """
        from langchain.prompts import ChatPromptTemplate
        short_answer_template = ChatPromptTemplate.from_template(
            """Given the following query: {query}
            and the following answer: {answer}
            extract from the answer short answer. Contain only answer in short answer, to not add additional text. Please always use relevant emojis to make the answer more visually appealing, to not inform that you have put emojis.
            """)
        short_answer_chain = short_answer_template | self.llm
        short_answer = short_answer_chain.invoke({'query': query, 'answer': answer})
        return short_answer.content
    
    def complex_answer(self, query: str, answer: str) -> str:
        """
        Based on the original prompt/query and the answer generated by the crew, it creates a lengthy more comples and detailed answer utilizing langchain chains. It's worth noting that this section is additionally structured in proper subsections, however, this structuring is managed fully by the LLM.
        """
        from langchain.prompts import ChatPromptTemplate
        complex_answer_template = ChatPromptTemplate.from_template(
            """Given the following query: {query}
            and the following answer: {answer}
            extract a complex well-structured explanation of the answer to the query in the clearly readable markdown format with some clear sections including sources if such where given or bulletpoints but don't add any headlines. You can just bold names of sections and add relevant emoji. But do not include any technical details like SQL code, database codes or tables names or any suggestions for visualization. You can add the tables if you think it's serves the purpose of building a good story. Try to be concise though if possible. Please always use relevant emojis to make the answer more visually appealing e.g. markdown tbale formatting. Do not inform that you have put emojis. prepare all the bulletpoints in a clear markdown format. Place sources at the end of the answer. Please remember to include both names and the links in the sources section
            """)
        complex_answer_chain = complex_answer_template | self.llm
        complex_answer = complex_answer_chain.invoke({'query': query, 'answer': answer})
        return complex_answer.content
    
    def data_chart_answer(self, query: str, answer: str) -> str:
        """
        Based on the original prompt/query and the answer generated by the crew, it extracts the data for visualization utilizing langchain chains. 
        """
        from langchain.prompts import ChatPromptTemplate
        data_chart_answer_template = ChatPromptTemplate.from_template(
            """Given the following query: {query}
            and the following answer: {answer}
            do the following
            If there is no relevant data in PIRLS database for visualization purpose to answer the query return only empty string: ''.
            If there is relevant data to answer the query, Extract the data for visualization from the response, and then create a well-labeled and clear plot using Matplotlib code.
            - Use only matplotlib and make sure all the data points used by matplotlib functions are created within this code snippet
            - Make it fancy, colourful, publication ready but extremely readable. Adjust font sizes but use standard fonts which are usually available. Avoid unnecesary lines especially boxes around the plot. Prepare a plot which makes people say AMAZING PLOT.
            - But the most important is prepare the code which actually works. Make sure the code works and can be executed.
            - Don't extract just one single number for visualization
            - Feel free to limit the data to top 10 or bottom 10 if you think it's better for visualization but make it clear in the title or labels that you selected best or worst groups e.g. countries
            - add a nice footnote with a source and align it to right
            - add subtitle if it's useful but make sure it does not overlay the title.
            - whenever you have 2 different metrics on the barchart which have different scales, use 2 different y axes!
            - add legend if the plot is complex.
            - always add labels with well formatted numbers or percentages
            - make sure there is a good title, x and y axis label and everything is clearly presented. AVOID overlap of text.
            - sort top values in descending order and lowest values in ascending
            - if there are multiple tables just choose one most relevant and visualize it. Don't overcompliucate things.
            - Always name chart in variable: plt
            - Do not use plt.show(), do not add plt.show() to this code. Do not include any other line of code which actually displays the plot. plt must be created and must not be displayed 
            Provide only the python code as the response, to not add additional text. Keep formating in best programming practices. Each new operation start from new line of code.
            """)
        data_chart_answer_chain = data_chart_answer_template | self.llm
        data_chart_answer = data_chart_answer_chain.invoke({'query': query, 'answer': answer})
        return data_chart_answer.content
    
    def extract_markdown_data_scientist(self, answer: str) -> str:
        """
        Based on the original prompt/query and the answer generated by the crew, it extracts the markdown for visualization purposes utilizing langchain chains. This markdown is an exact test which includes the visualization using S3 link.
        """
        from langchain.prompts import ChatPromptTemplate
        extract_markdown_template = ChatPromptTemplate.from_template(
            """Given the {answer}
            extract only the markdown part used for visualization. Do not change anything. Just extract it. dont add anything This must be a working markdown.
            If there is nothing relevant just return empty string ''
            """)
        extract_markdown_chain = extract_markdown_template | self.llm
        markdown = extract_markdown_chain.invoke({'answer': answer})
        return markdown.content
    
    def dad_joke(self, query: str, answer: str) -> str:
        """
        Based on the original prompt/query and the answer generated by the crew, it creates a so-called dad joke. 
        """
        from langchain.prompts import ChatPromptTemplate
        dad_joke_answer_template = ChatPromptTemplate.from_template(
            """Given the following query: {query}
            and the following answer: {answer}
            provide a dad joke relevant to this query and answer. Be creative but stick to the topic. Be funny. Use emojis and format it in markdown with some styling but don't use big headlines.
            Provide only the content of the joke with styling without any additional text without repeating the answer, without anything which is not your dad joke.
            """)
        dad_joke_answer_chain = dad_joke_answer_template | self.llm
        dad_joke_answer = dad_joke_answer_chain.invoke({'query': query, 'answer': answer})
        return dad_joke_answer.content

    def random_string(self, length: int) -> str:
        """
        The goal of the function is to create a string of random characters of given length. It's used for the purpose of creating new visualization which should be unique.
        """
        import random
        import string
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    def make_a_chart(self, code: str) -> str:
        """
        Based on the code which is supposed to make a plot and is generated by other function, this function executes the code and uploads the plot to S3.
        It returns URL to the uploaded plot so that it can be integrated in Markdown.
        """
        import boto3.session
        import matplotlib.pyplot as plt
        import boto3
        import io

        exec(code)
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=300, bbox_inches='tight')
        plt.close()

        filename = self.random_string(20)

        # Reset the pointer of the BytesIO object
        img_data.seek(0)
        

        # Upload the image to S3
        session = boto3.Session()
        s3 = session.client('s3')
        bucket_name = 'gdsc-bucket-058264313357'
        s3.upload_fileobj(img_data, bucket_name, filename)
        # Build and return the S3 URL
        s3_url = f'https://{bucket_name}.s3.amazonaws.com/{filename}'
        return s3_url
    
    @agent
    def lead_data_analyst(self) -> Agent:
        """
        Defines lead data analyst agent
        """
        a = Agent(
            config=self.agents_config['lead_data_analyst'],
            llm=self.llm,
            allow_delegation=True,
            verbose=True
        )
        return a

    @agent
    def data_engineer(self) -> Agent:
        """
        Defines data engineering agent
        """
        a = Agent(
            config=self.agents_config['data_engineer'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_execution_time = 300,
            max_retry_limit = 2,
            max_iter = 10,
            tools=[
                db_tools.query_database,
                db_tools.get_answers_to_question,
                research_tools.duckduckgo_tool
            ]
        )
        return a
    
    @agent
    def chart_preparer(self) -> Agent:
        """
        Defines chart preparer agent
        """
        a = Agent(
            config=self.agents_config['chart_preparer'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_execution_time = 180,
            tools=[
                db_tools.query_database
            ]
        )
        return a
    
    @agent
    def data_scientist(self) -> Agent:
        """
        Defines data scientist agent
        """
        a = Agent(
            config=self.agents_config['data_scientist'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_execution_time = 180
        )
        return a

    @task
    def data_science_task(self) -> Task:
        """
        Defines data science task
        """
        t = Task(
            config=self.tasks_config['data_science_task'],
            agent=self.data_scientist()
        )
        return t
    
    @task
    def answer_question_task(self) -> Task:
        """
        Defines main task which is to answer the question
        """
        t = Task(
            config=self.tasks_config['answer_question_task'],
            agent=self.lead_data_analyst()
        )
        return t

    @crew
    def crew(self) -> Crew:
        """Creates the data analyst crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            max_iter=2,
            max_execution_time = 300,
            cache=True
        )
