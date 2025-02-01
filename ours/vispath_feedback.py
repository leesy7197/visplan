import asyncio
import re
import os
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

class VisPathFeedback:
    def __init__(self, api_key=None, model="gpt-4o-mini", system_prompt_expansion=None, system_prompt_codegen=None, template_expansion=None, template_codegen=None, system_prompt_aggregation=None, template_aggregation=None):
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
        self.system_prompt_aggregation=system_prompt_aggregation or "You are an expert on data visualization code judgement and aggregation."

        self.template_aggregation=template_aggregation or """\
                Tink step by step, plan before think. Given the provided user query, data description, several data visualization codes generated for the query, and feedback of image generated for each code, generate a final version of code to fullfill the user query. You should write the generated code in the format of ```...```, where ... indicates the generated code.
                User Query: {ori_query}

                Data Description: {data_description}

                Code for aggregation with corresponding feedback: {code_for_aggregation}
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

    def describe_data_list(self, data_path_list):
        if data_path_list:
            data_description_list = []
            for data_path in data_path_list:
                single_data_description = self.describe_data(data_path)
                data_description_list.append(single_data_description)
            data_description = "[" + "], [".join(data_description_list) + "]"
        else :
            data_description = 'There is no dataset provided, please generate code based on the content of query.'

        return data_description

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

    async def single_codegen(self, query, data_description):
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

    async def codegen_agent(self, ori_query, extend_query_list, data_description):
        tasks = [self.single_codegen(query, data_description) for query in extend_query_list]

        generated_code_list = await asyncio.gather(*tasks)

        ori_generated_code = await self.single_codegen(ori_query, data_description)

        generated_code_list.append(ori_generated_code)

        return generated_code_list

    # Function to execute the code and save the plot as an image
    def code_to_image(code, img_path):
        exec_globals = {"plt": plt, "io": io}  # Define the global context for the code execution
        exec_locals = {}  # Local context for the code execution
        code = code.replace("plt.show()", f"plt.savefig('{img_path}')")  # Replace plt.show() with savefig to save the image
        try:
            exec(code, exec_globals, exec_locals)  # Execute the code
            return True  # Return True if the code executed successfully
        except Exception as e:
            return False  # Return False if an error occurred during code execution

    # Function to encode the image to base64 format
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    async def visual_feedback(self, ori_query, code):
		# Save the plot as an image using the provided code
		success = code_to_image(code, img_path)
		if not success:
			return "Error in saving the image."  # Return an error message if image saving fails
	
		# Encode the image to base64 for sending as a message
		base64_image = encode_image(img_path)
		
		# Prepare the information to be sent to GPT-4o for feedback analysis
		information = {
			'query': ori_query,
			'code': code
		}
	
		# Prepare the messages for GPT-4o, including the system prompt, user prompt with code and query, and the image
		messages = [
			{"role": "system", 
			"content": '''Given a piece of code, a user query, and an image of the current plot, please determine whether the plot has faithfully followed the user query. Your task is to provide instruction to make sure the plot has strictly completed the requirements of the query. Please output a detailed step by step instruction on how to use python code to enhance the plot.'''
			},
			{"role": "user", 
			"content": '''Here is the code: [Code]:
	"""
	{{code}}
	"""
	
	Here is the user query: [Query]:
	"""
	{{query}}
	"""
	
	Carefully read and analyze the user query to understand the specific requirements. Examine the provided Python code to understand how the current plot is generated. Check if the code aligns with the user query in terms of data selection, plot type, and any specific customization. Look at the provided image of the plot. Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. Compare these elements with the requirements specified in the user query. Note any differences between the user query requirements and the current plot. Based on the identified discrepancies, provide step-by-step instructions on how to modify the Python code to meet the user query requirements. Suggest improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements.'''}
		]
		messages[-1]["content"] += f"\n\n![plot](data:image/png;base64,{base64_image})"
		# Call the completion function to get feedback from GPT-4
		feedback = await self._call_openai_api(messages[0]['content'], messages[-1]['content'])

        return feedback

    async def feedback_aggregation(self, ori_query, generated_code_list):
        tasks = [self.visual_feedback(ori_query, code) for code in generated_code_list]

        generated_feedback_list = await asyncio.gather(*tasks)

        code_with_feedback = ""
        for i in range(len(generated_code_list)):
            code_with_feedback += f"------\ncode{i+1}:\n{generated_code_list[i]}\nfeedback{i+1}:\n{generated_feedback_list[i]}\n"

        return code_with_feedback


    async def get_code_content(self, ori_query, data_path_list):
        data_description = self.describe_data_list(data_path_list)
        extend_query_list = await self.expansion_agent(ori_query)
        generated_code_list = await self.codegen_agent(extend_query_list, data_description)
        code_for_aggregation = await self.feedback_aggregation(ori_query, generated_code_list)

        prompt = self.template_aggregation.format(
                ori_query=ori_query,
                data_description=data_description,
                code_for_aggregation=code_for_aggregation
                )

        response_text = await self.call_openai_api(self.system_prompt_aggregation, prompt)
        match = re.search(r"```(.*?)```", response_text, flags=re.DOTALL)
        if match:
            code_content = match.group(1).strip()
            return code_content
        else:
            return None


