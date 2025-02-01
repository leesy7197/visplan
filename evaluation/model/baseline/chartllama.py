import asyncio
import re
import os
import torch
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChartLlama:
    def __init__(self, model="listen2you002/ChartLlama-13b", system_prompt=None, data_path=None):
        # Use environment variables if no parameters are provided
        self.model = model
        self.system_prompt = system_prompt or "Think step by step. Based on the user's query, generate Python code using only `matplotlib.pyplot` to create the requested plot. Ensure the code is outputted within the format {code}...{/code}."
        self.data_path = data_path

        # Initialize Hugging Face pipeline
        self.client = pipeline("text-generation", model=self.model)

    def describe_data(self):
        """Generate description of the dataset (column names, data types, shape, etc.)"""
        if not self.data_path:
            return "No data file provided."
        
        try:
            # Read the data file using pandas
            data = pd.read_csv(self.data_path)
            description = {
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "shape": data.shape,
            }
            return description
        except Exception as e:
            return f"Error reading data file: {e}"

    async def call_chartllama_api(self, user_query, data_description):
        # Retry indefinitely until successful
        while True:
            try:
                # Concatenate the data description and user query
                full_query = f"Data Description: {data_description}\nUser Query: {user_query}"
                
                # Call ChartLlama API to get the response
                response = self.client(full_query, max_length=500)

                # Extract the code content from the response
                response_text = response[0]['generated_text']
                match = re.search(r"\{code\}(.*?)\{/code\}", response_text, flags=re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    return None
            except Exception as e:
                print(f"API call failed with error: {e}. Retrying...")

    async def get_code_content(self, user_query):
        # Get data description
        data_description = self.describe_data()

        # Call API and return the generated code
        code_content = await self.call_chartllama_api(user_query, data_description)
        return code_content