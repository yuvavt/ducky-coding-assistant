import os
import asyncio

from autogen import ConversableAgent
from aitools_autogen import utils
from aitools_autogen.agents import WebPageScraperAgent
from aitools_autogen.blueprint import Blueprint
from aitools_autogen.config import llm_config_openai as llm_config, config_list_openai as config_list, WORKING_DIR


class CodeChefCodingBlueprint(Blueprint):

    def __init__(self, work_dir: str = WORKING_DIR):
        super().__init__([], config_list=config_list, llm_config=llm_config)
        self._work_dir = work_dir
        self._solution_result = None

    @property
    def solution_result(self):
        return self._solution_result

    async def initiate_work(self, url: str):
        brute = os.path.join(self._work_dir, "brute")
        optimal = os.path.join(self._work_dir, "optimal")

        # Create directory if it doesn't exist
        os.makedirs(brute, exist_ok=True)
        os.makedirs(optimal, exist_ok=True)

        agent0 = ConversableAgent("a0",
                                  max_consecutive_auto_reply=0,
                                  llm_config=False,
                                  human_input_mode="NEVER")

        scraper_agent = WebPageScraperAgent()

        agent0.initiate_chat(scraper_agent, True, True, message=url)
        message = agent0.last_message(scraper_agent)["content"]

        # First LLM agent for generating brute force solution in Python
        brute_agent = ConversableAgent("brute_agent",
                                        max_consecutive_auto_reply=6,
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",
                                        code_execution_config=False,
                                        function_map=None,
                                        system_message="""You are a coding expert tasked with generating a brute force Python code solution for the given CodeChef problem URL.
        Please provide a brute force Python solution code for the given problem URL.
        """)

        agent0.initiate_chat(brute_agent, True, True, message=message)
        brute_solution_message = agent0.last_message(brute_agent)
        brute_solution = brute_solution_message["content"]

        # Second LLM agent for generating optimized solution in Python
        optimal_agent = ConversableAgent("optimal_agent",
                                         max_consecutive_auto_reply=6,
                                         llm_config=llm_config,
                                         human_input_mode="NEVER",
                                         code_execution_config=False,
                                         function_map=None,
                                         system_message="""You are a coding expert tasked with generating an optimal Python code solution for the given CodeChef problem URL.
        Please provide an optimal Python solution code for the given problem URL.
        """)

        agent0.initiate_chat(optimal_agent, True, True, message=message)
        optimal_solution_message = agent0.last_message(optimal_agent)
        optimal_solution = optimal_solution_message["content"]

        # Save both solutions to the respective directories
        utils.save_code_files(brute_solution, brute)
        utils.save_code_files(optimal_solution, optimal)

        self._solution_result = f"Brute Force Python Solution:\n{brute_solution}\n\nOptimal Python Solution:\n{optimal_solution}"


async def main(url):

    # Create an instance of the CodeChefCodingBlueprint
    blueprint = CodeChefCodingBlueprint()

    # Run the blueprint asynchronously to initiate work
    await blueprint.initiate_work(url)

    # Access the solution result after initiating work
    solution_result = blueprint.solution_result
    print(solution_result)


if __name__ == "__main__":
    # Pass the URL as an argument to the main function
    asyncio.run(main(url="https://www.codechef.com/practice/course/linked-lists/LINKLISTP/problems/INSDELDLL"))
