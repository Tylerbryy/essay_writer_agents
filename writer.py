import os
from crewai import Agent, Task, Crew

from langchain_openai import ChatOpenAI
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun


topic = input("Enter your essay topic: ")

from crewai import Task
from textwrap import dedent

search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


os.environ["SERP_API_KEY"] = "ff1899813e61be8cf317c34efe346542ec50f5ad59d56b80c0d57e1f0cc47306"
gpt4_preview= ChatOpenAI(temperature=0, openai_api_key="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", model="gpt-4-turbo-2024-04-09")


google_scholar_tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())



class EssayWritingTasks():
    def research(self, agent, topic):
        return Task(description=dedent(f"""
            Conduct comprehensive research on the topic: '{topic}'.
            Gather relevant information, data, and scholarly articles that provide depth and insight into the topic.
            Pay special attention to recent studies, expert opinions, and significant developments related to the topic.
            
            Your final output MUST be a detailed report summarizing the key findings, important statistics, and notable viewpoints on the topic.
            This report will serve as the foundation for the essay.
            
            Selected topic by the customer: {topic}
            {self.__tip_section()}
        """),
        agent=agent
        )

    def draft_essay(self, agent):
        return Task(description=dedent(f"""
            Based on the research conducted, draft an essay that is insightful, well-structured, and engaging.
            Ensure the essay has a clear introduction, body, and conclusion. Use the research report to support your arguments and provide evidence.
            
            Your draft MUST critically analyze the topic, presenting a balanced view and incorporating different perspectives.
            Make sure to cite all sources appropriately.
            
            {self.__tip_section()}
        """),
        agent=agent
        )

    def peer_review(self, agent):
        return Task(description=dedent(f"""
            Review the draft essay for coherence, academic integrity, and the relevance of its examples.
            Provide constructive feedback to enhance the content, focusing on clarity, argument strength, and overall flow.
            
            Your review MUST identify areas for improvement and suggest specific changes to refine the essay.
            Also, verify the accuracy of cited information and ensure that the essay meets academic standards.
            
            {self.__tip_section()}        
        """),
        agent=agent
        )

    def finalize_essay(self, agent):
        return Task(description=dedent(f"""
            Incorporate the feedback received during the peer review to refine and finalize the essay.
            Ensure that the essay is polished, well-formatted, and ready for submission.
            
            The final draft of the essay MUST incorporate all feedback received, be free of grammatical errors, and articulate a persuasive argument on the subject matter. It should captivate and educate the audience, ensuring an enriching reading experience.
            
            {self.__tip_section()}
        """),
        agent=agent
        )

    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"
    

    def enhance_essay(self, agent):
        return Task(description=dedent(f"""
            Refine the essay to make it more original and human-like. Increase the diversity of vocabulary and sentence structure, incorporate idiomatic expressions, and add personal anecdotes or unique examples where appropriate.
            
            Your enhancements should make the essay less detectable by AI detectors while maintaining or improving its readability and engagement.
                                       
            your answer MUST be the enhanced essay NOTHING ELSE, Make it pretty and well formatted.               List your sources!!!!!!!!!!!!!!!!!!!!             
            
            {self.__tip_section()}
        """),
        agent=agent
        )
    

# Create a Researcher agent with enhanced capabilities
researcher = Agent(
  role='Researcher',
  goal='Conduct exhaustive and meticulous research on the specified topic, ensuring a deep understanding and uncovering of both well-known facts and obscure details. Aim to provide a comprehensive foundation that supports a nuanced and insightful exploration of the subject matter.',
  verbose=True,
  llm=gpt4_preview,
  tools=[search_tool, google_scholar_tool],
  backstory='An AI meticulously trained in both academic and web research methodologies, capable of navigating through vast amounts of data to extract relevant and impactful information. This agent combines the precision of scholarly research with the breadth of web exploration to offer a rich, multi-dimensional view of any topic.',
)

# Create a Writer agent with enhanced capabilities
writer = Agent(
  role='Writer',
  goal='Draft an insightful, well-structured, and compelling essay that not only presents information but also engages and persuades the reader. Focus on creating a narrative that is rich in content, coherent in its argumentation, and elegant in its presentation, ensuring that the essay stands out for its depth of insight and clarity of expression.',
  verbose=True,
  tools=[search_tool],
  llm=gpt4_preview,
  backstory='An AI with unparalleled expertise in crafting engaging narratives and coherent arguments. This agent excels in transforming complex information into captivating stories, making sophisticated topics accessible and interesting to a broad audience. With a keen eye for detail and a masterful command of language, it ensures that every essay is a compelling read.',
)

enhancer = Agent(
  role='Enhancer',
  goal='Refine the essay to increase originality and reduce detectability by AI detectors.',
  verbose=True,
  llm=gpt4_preview,
  backstory='An AI skilled in enhancing the creativity and human-like quality of written content.',
)



essay_tasks = EssayWritingTasks()

research_task = essay_tasks.research(researcher, topic)
draft_task = essay_tasks.draft_essay(writer)
enhance_task= essay_tasks.enhance_essay(enhancer)



essay_crew = Crew(
  agents=[researcher, writer, enhancer], 
  tasks=[research_task, draft_task, enhance_task],  
)

# Begin the task execution
result =essay_crew.kickoff()
from colorama import Fore, Style

print(Fore.GREEN + "####################" + Style.RESET_ALL)
print(Fore.YELLOW + str(result) + Style.RESET_ALL)
