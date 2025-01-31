import asyncio
import re
import os
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

QUERY_EXPANSION_PROMPT='''According to the user query, expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. Import the appropriate libraries. Pinpoint the correct library functions to call and set each parameter in every function call accordingly.'''

SYSTEM_PROMPT_CODE_GEN="""You are a helpful assistant that generates Python code for data visualization and analysis using matplotlib and pandas. Given a detailed instruction and data description, generate the appropriate code in the format of {code}...{/code}'''

class MatplotAgent:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = api_key or os.getenv("API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def _call_openai_api(self, system_prompt, user_content):
        while True:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API call failed with error: {e}. Retrying...")

    async def _query_extension(self, nl_query):
        return await self._call_openai_api(QUERY_EXPANSION_PROMPT, nl_query)

    def _describe_data(self, data_path):
        if not data_path:
            return "No data file provided."
        try:
            data = pd.read_csv(data_path)
            return {
                "columns": list(data.columns),
                "dtypes": data.dtypes.apply(lambda x: x.name).to_dict(),
                "shape": data.shape,
                "sample": data.head(3).to_dict()
            }
        except Exception as e:
            return f"Error reading data file: {e}"

    async def generate_code(self, nl_query, data_path):
        data_description = self._describe_data(data_path)
        extended_query = await self._query_extension(nl_query)
        
        user_content = f"""Detailed Instructions:
{extended_query}

Data Description:
{data_description}

Please generate Python code that follows these instructions using the dataframe df."""

        response_text = await self._call_openai_api(SYSTEM_PROMPT_CODE_GEN, user_content)
        match = re.search(r"\{code\}(.*?)\{/code\}", response_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

