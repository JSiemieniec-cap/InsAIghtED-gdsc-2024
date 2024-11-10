# %%
#This file contains tests covering the main functionality.

# %%
!pip install -r ../requirements.txt

# %%
!pip install pytest

# %%
#pip install packaging

# %%
import dotenv
dotenv.load_dotenv()

# %%
import os
import sys 

sys.path.append('../')
os.getcwd()

# %%
from src.submission import create_submission
crew = create_submission.create_submission(12321)

# %%
import crewai

assert type(crew.crew()) == crewai.crew.Crew

assert '../src/submission/config/agents_rag_gdp.yaml' in str(crew.original_agents_config_path)
assert '../src/submission/config/tasks_rag_gdp.yaml' in str(crew.original_tasks_config_path)

agent_config = crew.agents_config

assert crew.is_crew_class

assert len(crew.agents) == 4
assert 'lead_data_analyst' in agent_config.keys()
assert 'data_engineer' in agent_config.keys()
assert 'chart_preparer' in agent_config.keys()
assert 'data_scientist' in agent_config.keys()

assert len(crew.tasks) == 2
assert 'data_science_task' in crew.tasks_config.keys()
assert 'answer_question_task' in crew.tasks_config.keys()

assert 'role' in agent_config['lead_data_analyst'] and 'goal' in agent_config['lead_data_analyst'] and 'backstory' in agent_config['lead_data_analyst']
assert 'role' in agent_config['data_engineer'] and 'goal' in agent_config['data_engineer'] and 'backstory' in agent_config['data_engineer']
assert 'role' in agent_config['chart_preparer'] and 'goal' in agent_config['chart_preparer'] and 'backstory' in agent_config['chart_preparer']
assert 'role' in agent_config['data_scientist'] and 'goal' in agent_config['data_scientist'] and 'backstory' in agent_config['data_scientist']

assert 'description' in crew.tasks_config['data_science_task'].keys() and 'expected_output' in crew.tasks_config['data_science_task'].keys()
assert 'description' in crew.tasks_config['answer_question_task'].keys() and 'expected_output' in crew.tasks_config['answer_question_task'].keys()


assert type(crew.lead_data_analyst()) == crewai.agent.Agent
assert type(crew.data_engineer()) == crewai.agent.Agent
assert type(crew.chart_preparer()) == crewai.agent.Agent
assert type(crew.data_scientist()) == crewai.agent.Agent

assert type(crew.data_science_task()) == crewai.task.Task
assert type(crew.answer_question_task()) == crewai.task.Task


# %%
query = 'Is it possible to cook a cake?'
answer = ''' Yes, you can absolutely cook a cake! In fact, baking a cake is one of the most common ways to make it. The process typically involves mixing ingredients like flour, sugar, eggs, butter or oil, and a leavening agent (like baking powder) to create a batter, then baking it in the oven at a controlled temperature.

However, "cooking" a cake can refer to some alternative methods, especially if you don’t have an oven. Here are a few options:

Microwave Cake: You can make mug cakes by combining a small portion of cake batter in a mug and microwaving it for a minute or so. Microwave cakes cook quickly but tend to be denser than oven-baked cakes.

Stovetop Cake: Using a heavy pan or a skillet, you can cook a cake on the stovetop. Often, a pot or a Dutch oven is used to simulate oven-like conditions. You'll need low, even heat and might need to flip the cake midway.

Steamed Cake: In some cuisines, cakes are steamed rather than baked. The batter is placed in a pan, which is then set in a steamer. This method results in a moist, soft cake and is often used for specific cake recipes, like Chinese sponge cakes.

Slow Cooker Cake: A slow cooker can also be used to “bake” a cake by cooking it on low heat over a long period, resulting in a soft, tender cake. This method works particularly well for dense cakes, like chocolate lava cake or pudding cake.

Each method has a slightly different result, but all are ways to "cook" a cake!'''

# %%
assert type(crew.short_answer(query,answer)) == str
assert type(crew.complex_answer(query,answer)) == str
# If there is no relevant data in PIRLS database for visualization purpose to answer the query return only empty string: ''.
assert crew.data_chart_answer(query, answer) == "''"
plot_code = crew.data_chart_answer('Compare revenues for companies: X = 125, Y = 250, Z = 155', answer = "Here's a comparison of the revenues for companies X, Y, and Z: Company Y has the highest revenue at 250. Company Z follows with a revenue of 155. Company X has the lowest revenue, at 125.")
assert "matplotlib" in plot_code
assert '.s3.amazonaws.com' in crew.make_a_chart(code = plot_code)
assert type(crew.dad_joke(query= 'How was your day?',answer = 'It was a good day')) == str
assert type(crew.random_string(20)) == str
assert len(crew.random_string(20)) == 20

assert "no markdown" in crew.extract_markdown_data_scientist(answer)


markdown = '''

> #### Short answer

# Company Revenue Comparison

This is a comparison of revenues for three companies: **Company X**, **Company Y**, and **Company Z**.

> #### Data visualization

!["Comparison"](https://my-bucket-12345.s3.amazonaws.com/comparison)
'''

assert crew.extract_markdown_data_scientist(markdown) == '!["Comparison"](https://my-bucket-12345.s3.amazonaws.com/comparison)'

# %%
answer = crew.run('Hello')
assert type(answer) == str and len(answer) > 0
#check if crew creates rag
assert os.path.exists('./rag')


