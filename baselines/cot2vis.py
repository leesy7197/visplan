import asyncio
import re
import os
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Cot2vis:
    def __init__(self, api_key=None, base_url=None, model="gpt-4o-mini", system_prompt=None, data_path=None):
        # Use environment variables if no parameters are provided
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model
        self.system_prompt = system_prompt or "Think step by step. Based on the user's query, generate Python code using only `matplotlib.pyplot` to create the requested plot. Ensure the code is outputted within the format {code}...{/code}."
        self.data_path = data_path

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=f'{self.base_url}/v1'
        )

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

    async def call_openai_api(self, user_query, data_description):
        # Retry indefinitely until successful
        while True:
            try:
                # Concatenate the data description and user query
                full_query = f"Data Description: {data_description}\nUser Query: {user_query}"
                
                # Call OpenAI API to get the response
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_query}
                    ]
                )
                
                # Extract the code content from the response
                response_text = response.choices[0].message.content
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
        code_content = await self.call_openai_api(user_query, data_description)
        return code_content

