
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from typing import Dict
import os

class FitnessProfile:
    def __init__(self, gender: str, age: int, weight: float, goal: str):
        self.gender = gender
        self.age = age
        self.weight = weight
        self.goal = goal

def get_workout_plan(profile: FitnessProfile) -> str:
    # Securely retrieve API key from environment variable
    api_key = os.getenv("LANGCHAIN_GROQ_API_KEY")
    if not api_key:
        raise ValueError("API key is not set. Please configure the environment variable LANGCHAIN_GROQ_API_KEY.")

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.1-70b-versatile",
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional fitness trainer. Create a personalized workout plan based on the user's profile.
        Include specific exercises, sets, reps, and any relevant advice. Keep it practical and achievable."""),
        ("human", """Create a workout plan for:
        Gender: {gender}
        Age: {age}
        Weight: {weight}kg
        Goal: {goal}
        """)
    ])

    # Format the prompt with user's profile
    formatted_prompt = prompt.format_messages(
        gender=profile.gender,
        age=profile.age,
        weight=profile.weight,
        goal=profile.goal
    )

    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    return response.content

# Streamlit UI
st.title("Personal AI Fitness Trainer")
st.write("Get a personalized workout plan crafted by AI! Provide your details below:")

# Collect user input
gender = st.radio("Gender", ("M", "F"), help="Select your gender.")
age = st.number_input("Age", min_value=1, max_value=120, step=1, help="Enter your age.")
weight = st.number_input("Weight (in kg)", min_value=1.0, max_value=200.0, step=0.1, help="Enter your weight in kilograms.")
goal = st.text_input("Fitness Goal", placeholder="e.g., weight loss, muscle gain, general fitness", help="Describe your fitness goal.")

# Generate the workout plan
if st.button("Generate Workout Plan"):
    if gender and age and weight and goal:
        user_profile = FitnessProfile(gender, age, weight, goal)

        with st.spinner("Generating your workout plan..."):
            try:
                workout_plan = get_workout_plan(user_profile)
                st.success("Your workout plan is ready!")
                st.write("### Workout Plan")
                st.write(workout_plan)
            except ValueError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please fill in all the details.")

