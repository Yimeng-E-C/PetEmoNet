import os
import math
import glob
import tqdm
import random
import json
from datetime import datetime
import numpy as np
import pandas as pd
import shutil

from config import *
from chatgpt import *

# support six types of inputs
def func_get_response(batch, emos, modality, sleeptime):
    if modality == 'image':
        response = get_image_emotion_batch(batch, emos, sleeptime)
    elif modality == 'evoke':
        response = get_evoke_emotion_batch(batch, emos, sleeptime)
    elif modality == 'micro':
        response = get_micro_emotion_batch(batch, emos, sleeptime)
    elif modality == 'video':
        response = get_video_emotion_batch(batch, emos, sleeptime)
    elif modality == 'text':
        response = get_text_emotion_batch(batch, emos, sleeptime)
    elif modality == 'multi':
        response = get_multi_emotion_batch(batch, emos, sleeptime)
    return response

# split one batch into multiple segments
def func_get_segment_batch(batch, savename, xishu=2):
    segment_num = math.ceil(len(batch)/xishu)
    store = []
    for ii in range(xishu):
        segbatch = batch[ii*segment_num:(ii+1)*segment_num]
        segsave  = savename[:-4] + f"_segment_{ii+1}.npz"
        if not isinstance(segbatch, list):
            segbatch = [segbatch]
        if len(segbatch) > 0:
            store.append((segbatch, segsave))
    return store

# main process
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, modality, bsize, xishus, batch_flag='flag1', sleeptime=0):
    # params assert to ensure process
    if len(xishus) == 1: assert batch_flag in ['flag1', 'flag2']
    if len(xishus) == 2: assert batch_flag in ['flag1', 'flag2', 'flag3']
    if len(xishus) == 3: assert batch_flag in ['flag1', 'flag2', 'flag3', 'flag4']
    multiple = 1
    for item in xishus: multiple *= item 
    assert multiple == bsize, f'multiple of xishus should equal to bsize'
    
    # Determine target folder for NPZs.
    # Always direct outputs into a folder that contains 'gpt4o' in its name.
    # - If the incoming save_root basename contains 'gpt4v', replace it with 'gpt4o' and replace '-' with '_'.
    # - Otherwise, normalize '-'->'_' and append '_gpt4o' if not already present.
    parent, base = os.path.split(save_root)
    candidate = base.replace('-', '_')
    if 'gpt4v' in candidate:
        candidate = candidate.replace('gpt4v', 'gpt4o')
    if 'gpt4o' not in candidate:
        candidate = candidate + '_gpt4o'
    save_root_used = os.path.join(parent, candidate)

    if not os.path.exists(save_root_used):
        os.makedirs(save_root_used, exist_ok=True)
    print(f"[INFO] Saving NPZs for this modality into: {save_root_used}")
    
    # preprocess for 'multi', multiple modal from a video, like text, audio and image
    if modality == 'multi':
        image_root = os.path.split(image_root)[0] + '/video'

    # shuffle image orders, if we have not generated a fixed npz for training order, we should get a new one
    if not os.path.exists(save_order):
        image_paths = glob.glob(image_root + '/*')
        indices = np.arange(len(image_paths))
        random.shuffle(indices) # fetch images randomly
        image_paths = np.array(image_paths)[indices]
        np.savez_compressed(save_order, image_paths=image_paths) # create npz to save image name
    else:
        image_paths = np.load(save_order, allow_pickle=True)['image_paths'].tolist()
    print (f'process sample numbers: {len(image_paths)}')

    # split int batch [20 samples per batch]
    batches = []
    splitnum = math.ceil(len(image_paths) / bsize) # split the whole dataset by bsize
    for ii in range(splitnum):
        batches.append(image_paths[ii*bsize:(ii+1)*bsize])
    print (f'process batch  number: {len(batches)}')
    print (f'process sample number: {sum([len(batch) for batch in batches])}')
    
    # generate predictions for each batch and store
    for ii, batch in tqdm.tqdm(enumerate(batches)):
        save_path = os.path.join(save_root_used, f'batch_{ii+1}.npz')
        if os.path.exists(save_path): continue
        ## batch not exists -> how to deal with these false batches
        if batch_flag == 'flag1': # process the whole batch again # 20
            response = func_get_response(batch, emos, modality, sleeptime)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
            # also save raw response as a text file for easier inspection
            try:
                txt_path = save_path[:-4] + '.txt'
                with open(txt_path, 'w', encoding='utf-8') as _f:
                    _f.write(response)
            except Exception:
                pass
        elif batch_flag == 'flag2': # split and process # 10
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = func_get_response(segbatch, emos, modality, sleeptime)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
                try:
                    txt_path = segsave[:-4] + '.txt'
                    with open(txt_path, 'w', encoding='utf-8') as _f:
                        _f.write(response)
                except Exception:
                    pass
        elif batch_flag == 'flag3': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = func_get_response(newbatch, emos, modality, sleeptime)
                    np.savez_compressed(newsave, gpt4v=response, names=newbatch)
                    try:
                        txt_path = newsave[:-4] + '.txt'
                        with open(txt_path, 'w', encoding='utf-8') as _f:
                            _f.write(response)
                    except Exception:
                        pass
        elif batch_flag == 'flag4': # split and process # 1
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    new2stores = func_get_segment_batch(newbatch, newsave, xishu=xishus[2])
                    for (new2batch, new2save) in new2stores:
                        if os.path.exists(new2save): continue
                        response = func_get_response(new2batch, emos, modality, sleeptime)
                        np.savez_compressed(new2save, gpt4v=response, names=new2batch)
                        try:
                            txt_path = new2save[:-4] + '.txt'
                            with open(txt_path, 'w', encoding='utf-8') as _f:
                                _f.write(response)
                        except Exception:
                            pass
    # return the actual folder used for this run so caller can parse it
    return save_root_used
                            
def func_analyze_gpt4v_outputs(gpt_path):
    names = np.load(gpt_path, allow_pickle=True)['names'].tolist()

    ## analyze gpt-4v
    store_results = []
    gpt4v = np.load(gpt_path, allow_pickle=True)['gpt4v'].tolist()
    gpt4v = gpt4v.replace("name",    "==========")
    gpt4v = gpt4v.replace("result",  "==========")
    gpt4v = gpt4v.split("==========")
    for line in gpt4v:
        if line.find('[') != -1:
            res = line.split('[')[1]
            res = res.split(']')[0]
            store_results.append(res)
    
    return names, store_results
    
def check_gpt4_performance(gpt4v_root):
    error_number = 0
    whole_names, whole_gpt4vs = [], []
    # only iterate over npz files; avoid parsing .txt debug files
    for gpt_path in sorted(glob.glob(os.path.join(gpt4v_root, '*.npz'))):
        names, gpt4vs = func_analyze_gpt4v_outputs(gpt_path)
        print (f'number of samples: {len(names)} number of results: {len(gpt4vs)}')
        if len(names) == len(gpt4vs): 
            names = [os.path.basename(name) for name in names]
            whole_names.extend(names)
            whole_gpt4vs.extend(gpt4vs)
        else:
            print (f'error batch: {gpt_path}. Need re-test!!')
            # Instead of outright deleting the problematic batch file, move it to a 'failed' folder
            # so the file can be inspected or recovered later.
            failed_dir = os.path.join(os.path.dirname(gpt_path), 'failed')
            os.makedirs(failed_dir, exist_ok=True)
            try:
                shutil.move(gpt_path, os.path.join(failed_dir, os.path.basename(gpt_path)))
                print(f'Moved failed batch to {failed_dir}')
            except Exception as _e:
                print(f'Failed to move {gpt_path} to {failed_dir}: {_e}. Skipping deletion.')
            error_number += 1
    print (f'error number: {error_number}')
    return whole_names, whole_gpt4vs


def func_parse_and_save_softlabels(gpt4v_root, out_csv, class_list=None):
    """Parse saved GPT responses (npz files) where each response is expected to be a JSON array
    of objects {"name":..., "probabilities": {label: prob, ...}}. Writes a CSV with columns:
    filename,prob_<label1>,prob_<label2>,...
    If parsing fails for a batch, it will be skipped with a warning.
    """
    records = []
    # only consider npz files (avoid .txt debug files saved alongside npz)
    files = sorted(glob.glob(os.path.join(gpt4v_root, '*.npz')))
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            names = data['names'].tolist()
            raw = data['gpt4v'].tolist()
            # raw expected to be JSON string
            try:
                parsed = json.loads(raw)
            except Exception:
                # try to extract JSON substring
                s = raw
                start = s.find('[')
                end = s.rfind(']')
                if start != -1 and end != -1:
                    try:
                        parsed = json.loads(s[start:end+1])
                    except Exception:
                        print(f"Warning: cannot parse JSON in {f}; skipping batch")
                        continue
                else:
                    print(f"Warning: no JSON array found in {f}; skipping batch")
                    continue

            if not isinstance(parsed, list):
                print(f"Warning: parsed content is not a list in {f}; skipping")
                continue

            for item in parsed:
                # accept both 'name' and 'filename' keys
                fname = item.get('name') or item.get('filename')
                probs = item.get('probabilities') or item.get('probs') or item.get('scores')
                if fname is None or probs is None:
                    print(f"Warning: missing fields in parsed item in {f}; skipping item")
                    continue
                # normalize keys if needed
                records.append({'filename': os.path.basename(fname), **{f'prob_{k}': float(v) for k, v in probs.items()}})

        except Exception as e:
            print(f"Error reading {f}: {e}; skipping")
            continue

    if len(records) == 0:
        print('No soft-label records parsed. Nothing to save.')
        return None

    df = pd.DataFrame.from_records(records)
    # ensure columns for all classes exist
    if class_list is not None:
        for cls in class_list:
            col = f'prob_{cls}'
            if col not in df.columns:
                df[col] = 0.0

    df = df[['filename'] + sorted([c for c in df.columns if c.startswith('prob_')])]
    df.to_csv(out_csv, index=False)
    print(f'Saved soft labels CSV to {out_csv}')
    return out_csv

if __name__ == '__main__':

    # -------------- defined by users --------------- #
    dataset = 'dog'  # dataset type
    # save_root = r'e:\BaiduNetdiskDownload\LLMemotion\LLMemotion\LLMemotion\GPTonly\dataset\dog_test'  # dataset root - 测试集(100张)
    save_root = r'e:\BaiduNetdiskDownload\LLMemotion\LLMemotion\LLMemotion\GPTonly\dataset\dog'  # 完整数据集(4000张)
    # ----------------------------------------------- #

    # please pre-defined dataset-related params in config.py
    emos = dataset2emos[dataset] # emotion classifications in your dataset, for example, there're four classifications in the dog dataset
    modalities = dataset2modality[dataset] # input modals, such as image, audio and text, but we only use image here
    for modality in modalities:
        bsize, xishus = modality2params[modality] # some parameters to train a specific modal, bsize is batchsize, xishus is training times

        # flags: request multiple times
        flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3'] # flags to determine a traning mode
        if len(xishus) == 3:
            flags.append('flag4')

        # process for each modality
        image_root = os.path.join(save_root, modality)  # the folder path of this modal
        gpt4v_root = os.path.join(save_root, 'gpt5mini')  # store gpt results - 改为gpt5mini文件夹
        save_order = os.path.join(save_root, f'{modality}-order-gpt5mini.npz') # ensure each request is in the same order, becasue we use a fixed bsize
        for flag in flags: # training in different modes
            gpt_used_root = evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, bsize, xishus, batch_flag=flag, sleeptime=4)  # 优化为4秒
            # check and clean false/partial predictions in the actual run folder
            check_gpt4_performance(gpt_used_root)

        # After all flags processed for this modality, try to parse saved GPT responses and export soft labels CSV
        out_csv = os.path.join(save_root, f'{modality}_soft_labels.csv')
        # parse from the latest created run folder if evaluate created one, otherwise use default
        parse_root = gpt_used_root if 'gpt_used_root' in locals() else gpt4v_root
        func_parse_and_save_softlabels(parse_root, out_csv, class_list=emos)
