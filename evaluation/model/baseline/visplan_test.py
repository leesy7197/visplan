import asyncio
from cot2vis import Cot2vis
from chat2vis import Chat2Vis
from chartllama import ChartLlama

async def main():
    # Initialize the Cot2vis instance
    # Here we set the path to the CSV data file; adjust to your actual filename/path
    chat2vis_instance = Chat2Vis(
        data_path="example_data.csv"
        # Optionally provide other parameters if needed, such as:
        # api_key="YOUR_API_KEY",
        # base_url="YOUR_BACKEND_URL",
        # model="gpt-4o-mini"
    )

    # chartllama_instance = ChartLlama(
    # data_path="example_data.csv"
    # # Optionally provide other parameters if needed, such as:
    # # api_key="YOUR_API_KEY",
    # # base_url="YOUR_BACKEND_URL",
    # # model="gpt-4o-mini"
    # )   

    # Define the user query for generating a visualization
    user_query = "Please create a bar chart comparing the 2021 and 2022 sales."

    # Fetch the generated Python plotting code using the query
    generated_code = await chat2vis_instance.get_code_content(user_query)

    # Print the result to verify the code
    print("Generated code based on the user query:")
    print(generated_code)

# Use asyncio.run() to execute the async function in Python 3.7+
if __name__ == "__main__":
    asyncio.run(main())
