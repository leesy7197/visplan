import asyncio
import pandas as pd
from openai import AsyncOpenAI

client = AsyncOpenAI(
	api_key="",
	base_url="/v1"
)

concurrency_limit=
input_file=""
output_file=""

async def call_openai_api(model, system_prompt, user_content):
	# Infinite retries until success
	while True:
		try:
			response = await client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_content}
				]
			)
			return response.choices[0].message.content
		except Exception as e:
			print(f"API call failed with error: {e}. Retrying...")
			
async def function1(text):
	return await call_openai_api(text)
			
async def process_row(row, sem=sem):
	async with sem:
		some_var = await asyncio.gather(
			function1(row[..]),
			function2(row[..]),
			...
		)
	...
	
async def process_dataframe(df, sem, batch_size):
	results = []
	# Batch processing
	for i in range(0, len(df), batch_size):
		batch_df = df.iloc[i:i+batch_size]
		tasks = [process_row(row) for _, row in batch_df.iterrows()]
		batch_results = await asyncio.gather(*tasks)
		results.extend(batch_results)
		
	df['...'] = [r[0] for r in results]
	df['...'] = [r[1] for r in results]
	...
	return df

async def main():
	# Initialize the semaphore in main().
	sem = asyncio.Semaphore(concurrency_limit)
	
	# Summarize main_df
	processed_df = await process_dataframe(main_df, sem=sem, batch_size=batch_size)
	
	# Save results
	processed_df.to_csv(output_file, index=False)
	print("Done. Result saved to: ", output_file)
	
if __name__ == "__main__":
	asyncio.run(main())