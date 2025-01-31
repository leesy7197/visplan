import asyncio
import re
import os
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Cot2vis:
    def __init__(self, api_key=None, model="gpt-4o-mini", system_prompt=None):
        # Use environment variables if no parameters are provided
        self.api_key = api_key or os.getenv("API_KEY")
        self.model = model
        self.system_prompt = system_prompt or "Think step by step. Based on the user's query, generate Python code using `matplotlib.pyplot` and 'seaborn' to create the requested plot. Ensure the code is outputted within the format ```...```."

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key
        )

    def describe_data(self, data_path):
        # Read the data file using pandas
        data = pd.read_csv(data_path)
        description = {
            "data_path": data_path,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "shape": data.shape,
        }
        return str(description)

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
                match = re.search(r"```(.*?)```", response_text, flags=re.DOTALL)
                if match:
                    return match.group(1).strip()
                else:
                    return None
            except Exception as e:
                print(f"API call failed with error: {e}. Retrying...")

    async def get_code_content(self, user_query, data_path_list):
        if data_path_list:
            data_description_list = []
            for data_path in data_path_list:
                single_data_description = describe_data(data_path)
                data_description_list.append(single_data_description)
            data_description = "[" + "], [".join(data_description_list) + "]"
        else :
            data_description = 'There is no dataset provided.'

        # Call API and return the generated code
        code_content = await self.call_openai_api(user_query, data_description)
        return code_content
