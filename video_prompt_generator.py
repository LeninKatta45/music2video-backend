from crewai import Crew, Task, Agent, LLM
# Initialize LLM for prompt generation
llm = LLM(
    model="groq/llama3-70b-8192",
    temperature=0.7,  # Slightly higher temperature for creativity
    max_tokens=1024,  # Limit tokens to make the output concise
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

# Define the Prompt Generation Agent
prompt_agent = Agent(
    llm=llm,
    role="Video Prompt Creator",
    goal=(
        "Create a concise and visually compelling text prompt that captures the "
        "emotion, mood, and story of the audio. The prompt should translate lyrics, themes, "
        "colors, and animations into a vivid visual narrative."
    ),
    backstory=(
        "This agent is a creative expert in visual storytelling, capable of transforming "
        "audio input into rich, imaginative video prompts. The goal is to ensure alignment "
        "with the essence of the music and lyrics while inspiring dynamic visuals."
    ),
      # Exclude tools for simplicity unless specific web lookups are required
    allow_delegation=False,
    verbose=1
)

# Adjusted Task Description and Expected Output
prompt_task = Task(
    description=(
        "Generate a text prompt for video creation based on the following audio input: {topic}. "
        "The prompt should be concise (under 150 words), align with the emotion and story conveyed by the audio, "
        "and describe visually striking scenes, transitions, and effects that match the rhythm and mood of the music."
    ),
    expected_output=(
        "A vivid and concise video-generation prompt (under 150 words) that integrates the audio's lyrics, "
        "emotion, theme, color palette, and animation style into a cohesive visual concept."
    ),
    output_file="The_Best_Prompt.txt",
    agent=prompt_agent
)

# Initialize Crew with the refined task
crew = Crew(agents=[prompt_agent], tasks=[prompt_task], verbose=1)

def generate_video_prompt(input_data):
    """
    Generate a concise video prompt based on the given input data.
    
    Args:
        input_data (dict): Dictionary containing audio description, emotion, lyrics, theme, colors, and animation.
    
    Returns:
        str: Generated video prompt.
    """
    output = crew.kickoff(inputs={'topic': input_data})
    print("GOT IT:.......")
    print(output)
    
    
    return output
