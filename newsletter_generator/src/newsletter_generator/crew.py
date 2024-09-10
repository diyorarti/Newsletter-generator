from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from newsletter_generator.tools.search import SearchAndContents, FindSimilar, GetContents
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from datetime import datetime
# Uncomment the following line to use an example of a custom tool
# from newsletter_generator.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class NewsletterGeneratorCrew:
    """NewsletterGenerator crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

	def llm(self):
        llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", max_tokens=4096)
        # llm = ChatGroq(model="llama3-70b-8192")
        # llm = ChatGroq(model="mixtral-8x7b-32768")
        # llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"))

        return llm

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[SearchAndContents(), FindSimilar(), GetContents()],
            verbose=True,
			llm=self.llm()
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config["editor"],
            verbose=True,
            tools=[SearchAndContents(), FindSimilar(), GetContents()],
			llm=self.llm()
        )

    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            allow_delegation=False,
			llm=self.llm()
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_research_task.md",
        )

    @task
    def edit_task(self) -> Task:
        return Task(
            config=self.tasks_config["edit_task"],
            agent=self.editor(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_edit_task.md",
        )

    @task
    def newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["newsletter_task"],
            agent=self.designer(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_newsletter_task.html",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NewsletterGenerator crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you want to use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
