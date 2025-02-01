import asyncio
import re
import os
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
import ast

# Load environment variables
load_dotenv()


class Visplan:
    def __init__(self, api_key=None, model="gpt-4o-mini", system_prompt_expansion=None, system_prompt_codegen=None, template_expansion=None, template_codegen=None):
        self.api_key = api_key or os.getenv("API_KEY")
        self.model = model

        self.client = AsyncOpenAI(
            api_key=self.api_key
        )

        self.system_prompt_expansion=system_prompt_expansion or "You are an expert on extend data visualization query. You should think step by step, and write the extended queries in the format of ```[query_text_1, query_text_2, query_text_3]```"

        self.template_expansion=template_expansion or """\
                Think step by step. Generate three distinct extended queries based on the given query. Each query should emphasize a different aspect or perspective of the original query. Present the queries in the following format: ```[query_text_1, query_text_2, query_text_3]```, where query_text_n refers to the generated queries.
                The original query is {ori_query}
                """

        self.system_prompt_codegen=system_prompt_codegen or "You are an expert on data visualization code generation. You should think step by step, and write the generated code in the format of ```...```, where ... indicates the generated code."

        self.template_codegen=template_codegen or """\
                Think step by step. Based on the user's query and data description, generate Python code using `matplotlib.pyplot` and 'seaborn' to create the requested plot. Ensure the code is outputted within the format ```...```, where ... indicates the generated code.

                User query: {query}

                Data description: {data_description}
                """

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

    async def call_openai_api(self, system_prompt, user_content):
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


    async def expansion_agent(self, ori_query):
        prompt = self.template_expansion.format(
                ori_query=ori_query,
                )
        response_text = await self.call_openai_api(self.system_prompt_expansion, prompt)
        match = re.search(r"```(.*?)```", response_text, flags=re.DOTALL)
        if match:
            generated_query = match.group(1).strip()
            try:
                parsed_list = ast.literal_eval(generated_query)

                if isinstance(parsed_list, list) and parsed_list:
                    return parsed_list
                else:
                    return None
            except (ValueError, SyntaxError):
                return None
        else:
            return None

    async def single_codegen(self, query, data_path_list):
        if data_path_list:
            data_description_list = []
            for data_path in data_path_list:
                single_data_description = self.describe_data(data_path)
                data_description_list.append(single_data_description)
            data_description = "[" + "], [".join(data_description_list) + "]"
        else :
            data_description = 'There is no dataset provided, please generate code based on the content of query.'

        # Call API and return the generated code
        prompt=self.template_codegen.format(
                query=query,
                data_description=data_description,
                )

        response_text = await self.call_openai_api(self.system_prompt_codegen, prompt)
        match = re.search(r"```(.*?)```", response_text, flags=re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            return code_content
        else:
            return None

    async def codegen_agent(self, extend_query_list, data_path_list):
        tasks = [self.single_codegen(query, data_path_list) for query in extend_query_list]

        generated_code_list = await asyncio.gather(*tasks)

        return generated_code_list

