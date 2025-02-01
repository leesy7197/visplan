import base64
import json
import sys
sys.path.insert(0, sys.path[0]+"/../")
from loguru import logger # type: ignore
import os
import re
import ast
import shutil
import glob
import asyncio
import io
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import image_to_base64, base64_to_image, code_to_image

from model.baseline import Cot2vis, Chat2Vis, MatplotAgent
from eval_api import gpt_4_evaluate, gpt_4v_evaluate
from openai import OpenAI
from config.openai import API_KEY, BASE_URL, temperature

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='matplotbench', required=True)
parser.add_argument('--baseline', type=str, default='baseline', required=True)
args = parser.parse_args()

BASELINE_MAPPING = {
    "cot2vis": Cot2vis(api_key=API_KEY),
    "chat2vis": Chat2Vis(api_key=API_KEY),
    "matplotagent": MatplotAgent(api_key=API_KEY),
    # "our_model" : our_model
}

def mainworkflow(test_sample_id, benchmark=args.benchmark, baseline = args.baseline):
    directory = f'./result/{benchmark}/{baseline}/example_{test_sample_id}'

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")

    # logger.basicConfig(level=logger.INFO, filename=f'{directory}/eval_4v_rollback.log', filemode='w',
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.benchmark == 'matplotbench':
        matplotagent_bench = pd.read_csv(r'.\dataset\matplotagent_data.csv')
        id = matplotagent_bench.loc[test_sample_id - 1,'id']
        # simple_instruction = matplotagent_bench.loc[test_sample_id - 1,'simple_instruction']
        expert_instruction = matplotagent_bench.loc[test_sample_id - 1,'expert_instruction']
        ground_truth = matplotagent_bench.loc[test_sample_id - 1,'img_bs64']
        user_query = expert_instruction ## Need to Select Simple or Expert
        data_path = None

    elif args.benchmark == 'viseval':
        viseval_bench = pd.read_csv(r'.\dataset\viseval_data.csv')
        id = viseval_bench.loc[test_sample_id,'id']
        user_query = ast.literal_eval(viseval_bench.loc[test_sample_id,'nl_queries'])[0] ## First one of nl_queries
        ground_truth = viseval_bench.loc[test_sample_id,'img_bs64']
        database = viseval_bench.loc[test_sample_id,'db_id']
        tables = viseval_bench.loc[test_sample_id,'irrelevant_tables']
        data_path = [f'./dataset/databases/{database}/{table}.csv' for table in ast.literal_eval(tables)]

    print(f"ID : {id}")
    print(f"User Query : {user_query}")
    print(f"Data Path : {data_path}")

    logger.info('=========Baseline Selection=========')
    model = BASELINE_MAPPING[baseline]
    print(model, type(model))

    # try:
    #     df = pd.read_csv(data_path[0])
    #     print(df.head(2))
    # except Exception as e:
    #     print(f'Error during read csv : {e}')

    logger.info('=========Generating Plot Code=========')
    if args.baseline == 'cot2vis':
        code = asyncio.run(model.get_code_content(user_query = user_query, data_path_list = (None if not data_path else data_path)))
    elif args.baseline == 'chat2vis':
        code = asyncio.run(model.get_code_content(nl_query = user_query, data_path_list = (None if not data_path else data_path)))
    elif args.baseline == 'matplotagent':
        code = asyncio.run(model.generate_code(nl_query = user_query, data_path_list = (None if not data_path else data_path)))

    # print(expert_instruction)
    print('#'*10, 'Generated Code', '#'*10)
    print(code)

    logger.info('=========Execute Code & Save Image=========')
    img_save_path = f'{directory}/{id}.png'
    log = code_to_image(code, id, img_save_path)
    print(img_save_path)

    if not log:
        id = str(id)
        scores = {
            "ID" : id,
            "Executable Score" : 0,
            "Code Score" : 0,
            "Plot Score" : 0
        }
        print(scores)
        logger.info('=========Save Score=========')
        # scores를 JSON으로 저장
        log_scores_path = f'./result/{benchmark}/{baseline}/example_{test_sample_id}/result_{benchmark}_{baseline}_{id}.json'
        with open(log_scores_path, 'w') as json_file:
            json.dump(scores, json_file, indent=4)
    else:
        logger.info('=========Image to Base64=========')
        generated_bs64 = image_to_base64(img_save_path)

        logger.info('=========Evaluate Code (Score)=========')
        code_score = gpt_4_evaluate(code, user_query, img_save_path) 
        if code_score == 0:
            executable_score = 0

        else:
            executable_score = 1
            
        pattern = r'.*?\[FINAL SCORE\]: (\d{1,3})'
        code_score_extract = re.findall(pattern, code_score, re.DOTALL)[0]
        print(code_score)
        # print(code_score_extract)
        # print(executable_score)
        # print(generated_bs64)

        logger.info('=========Evaluate Image (Score)=========')
        plot_score = gpt_4v_evaluate(ground_truth, generated_bs64)
        pattern = r'.*?\[FINAL SCORE\]: (\d{1,3})'
        plot_score_extract = re.findall(pattern, plot_score, re.DOTALL)[0]
        print(plot_score)

        print(f"Executable Score : {executable_score}")
        print(f"Code Score : {code_score_extract}")
        print(f"Plot Score : {plot_score_extract}")
        id = str(id)
        scores = {
            "ID" : id,
            "Executable Score" : executable_score,
            "Code Score" : code_score_extract,
            "Plot Score" : plot_score_extract
        }
        print(scores)
        logger.info('=========Save Score=========')
        # scores를 JSON으로 저장
        log_scores_path = f'./result/{benchmark}/{baseline}/example_{test_sample_id}/result_{benchmark}_{baseline}_{id}.json'
        with open(log_scores_path, 'w') as json_file:
            json.dump(scores, json_file, indent=4)

if __name__ == "__main__":
    # Get the number passed as an argument
    # idx = int(sys.argv[1])
    idx = 74
    mainworkflow(test_sample_id=idx, benchmark=args.benchmark, baseline = args.baseline)