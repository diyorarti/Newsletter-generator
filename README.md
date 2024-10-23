# Newsletter Generator
This project implements a Newsletter Generator powered by CREWAI and designed to streamline the creation of newsletters through a series of tasks executed by specialized AI agents. The system performs research, edits content, and compiles a newsletter using HTML templates. The project is inspired by a concept originally presented by Alejandro on his YouTube channel.

## Project structure
```bash
newsletter_generator/
│
├── .gitignore                        # Ignored files for Git
├── LICENSE                           # License information
├── README.md                         # Project documentation (this file)
├── poetry.lock                       # Poetry lock file for dependency management
├── pyproject.toml                    # Poetry project configuration
├── logs/                             # Log files for each task execution
│   └── <timestamp>_research_task.md  # Research task logs
│   └── <timestamp>_edit_task.md      # Edit task logs
│   └── <timestamp>_newsletter_task.html  # Newsletter HTML output
│
├── src/
│   └── newsletter_generator/
│       └── config/                   # Configuration files for agents and tasks
│           └── agents.yaml           # YAML configuration for agents (Researcher, Editor, Designer)
│           └── tasks.yaml            # YAML configuration for tasks (Research, Edit, Newsletter generation)
│           └── newsletter_template.html  # HTML template for the newsletter layout
│       └── gui/                      # Streamlit-based web interface
│           └── app.py                # Streamlit app for generating newsletters via the web
│       └── tools/                    # Custom tools for web scraping and content extraction
│           └── search.py             # Tools for searching and fetching content using Exa API
│       └── crew.py                   # CREWAI-based agents and tasks execution
│       └── main.py                   # Command-line interface for generating newsletters
│
├── tests/                            # Test files for different modules
│   └── test.html                     # Test HTML file for newsletter generation
│
└── requirements.txt                  # Python dependencies for the project
```

## Features
- **Automated Newsletter Generation**: The system automates newsletter creation by performing web searches, curating articles, editing content, and generating an HTML newsletter.
- **Multi-Agent System**: Utilizes CREWAI to define roles for the Researcher, Editor, and Designer agents.
- **Flexible Input**: Users can provide topics and personal messages to tailor newsletters.
- **HTML Template**: A customizable HTML template is used to compile the final newsletter.
- **Sequential Task Execution**: Tasks such as researching, editing, and compiling the newsletter are executed sequentially.
- **Inspired by Alejandro’s YouTube tutorial**: The project is a learning exercise that dives into new technologies, inspired by Alejandro’s video content.

## Installation
1. Clone the Repository:
```bash
git clone https://github.com/yourusername/newsletter_generator.git
cd newsletter_generator
```
2. Install Dependencies:
Use Poetry to install the required packages:
```bash
poetry install
```
3.Set Environment Variables:
Create a .env file in the root directory and add your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
EXA_API_KEY=your_exa_api_key
```

## Usage
**Streamlit GUI**
You can run the Streamlit app to generate newsletters via a web interface:
```bash
streamlit run src/newsletter_generator/gui/app.py
```
- Enter the topic for the newsletter and a personal message to include.
- The generated HTML file will be downloadable once the process is complete.

# Agents and Tasks
The Newsletter Generator operates by delegating tasks to different agents, each responsible for a specific part of the newsletter creation process.

## Researcher Agent
- Role: Senior Researcher
- Goal: Uncover the most relevant news stories for the given topic using web search.
- Tools Used: Web search, content extraction.
- Task: Perform thorough research and summarize relevant articles.
## Editor Agent
- Role: Editor-in-Chief
- Goal: Review, edit, and reorder the news articles for maximum impact and relevance.
- Task: Rewrites titles, provides context for each article, and ensures the list is well-ordered.
## Designer Agent
- Role: Newsletter Compiler
- Goal: Compile the edited news into an HTML template without modifying the content.
- Task: Fills the provided HTML template with the curated news content.

## Configuration
The project is highly configurable, allowing easy customization of agents, tasks, and the newsletter template.

## Agent Definitions
Agents are defined in the agents.yaml file:
- Researcher: Gathers relevant news articles.
- Editor: Edits and reorders the articles for readability and engagement.
- Designer: Compiles the HTML newsletter.

## Task Definitions
Tasks are defined in the tasks.yaml file:
- Research Task: Defines the guidelines for researching news on the given topic.
- Edit Task: Ensures the news articles are properly edited and ordered.
- Newsletter Task: Compiles the final HTML newsletter.