import dotenv

from src.static.ChatBedrockWrapper import ChatBedrockWrapper
from src.static.submission import Submission
from src.submission.crews.advanced_PIRLS_crew_rag_gdp import AdvancedPIRLSCrew

dotenv.load_dotenv()


# This function is used to run evaluation of your model.
# You MUST NOT change the signature of this function! The name of the function, name of the arguments,
# number of the arguments and the returned type mustn't be changed.
# You can modify only the body of this function so that it returned your implementation of the Submission class.
def create_submission(call_id: str) -> Submission:
   
    llm = ChatBedrockWrapper(
        model_id= 'anthropic.claude-3-5-sonnet-20240620-v1:0',
        model_kwargs={'temperature': 0},
        call_id=call_id
    )

    crew = AdvancedPIRLSCrew(llm=llm)
    return crew
