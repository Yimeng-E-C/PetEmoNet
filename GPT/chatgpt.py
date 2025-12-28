import os
import cv2
import time
import glob
import base64
import numpy as np
from openai import OpenAI
from config import candidate_keys


global_index = 0
client = OpenAI(api_key=candidate_keys[global_index])


def _build_inputs(prompt):
    """Convert legacy prompt structure into Responses API payload."""
    if isinstance(prompt, list):
        content = []
        for item in prompt:
            if item.get('type') == 'text':
                content.append({"type": "input_text", "text": item['text']})
            elif 'image' in item:
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{item['image']}",
                })
            else:
                content.append({"type": "input_text", "text": str(item)})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": [{"type": "input_text", "text": str(prompt)}]}]


def func_get_completion(prompt, model="gpt-5-mini"):
    """Call GPT model (supports multimodal via Chat Completions API).
    Default model updated from gpt-4o-mini to gpt-5-mini.
    """
    global client, global_index
    try:
        # 构建消息格式
        if isinstance(prompt, list):
            content = []
            for item in prompt:
                if item.get('type') == 'text':
                    content.append({"type": "text", "text": item['text']})
                elif 'image' in item:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{item['image']}"},
                    })
                else:
                    content.append({"type": "text", "text": str(item)})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        
        # Some newer models (e.g. GPT-5 family) expect 'max_completion_tokens'
        # instead of the older 'max_tokens' parameter. Try the newer name
        # first for compatibility, and fall back if the client wrapper requires the old name.
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=1000,
                temperature=0,
            )
        except Exception as e_fallback:
            # Fallback: retry without explicit token-limit parameter. Some newer
            # models/clients reject the legacy `max_tokens` name while older
            # clients may reject `max_completion_tokens`. To avoid the
            # "Unsupported parameter: 'max_tokens'" server error, try a
            # parameter-free call as a last resort.
            try:
                # Retry without explicit temperature so the model's default
                # behaviour is used (avoids models that reject temperature=0).
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            except Exception:
                # Re-raise the original fallback exception so outer handler
                # can rotate API keys / log properly.
                raise e_fallback
        return response.choices[0].message.content
    except Exception as e:
        print('Error:', e)
        global_index = (global_index + 1) % len(candidate_keys)
        print(f'========== key index: {global_index} ==========' )
        client = OpenAI(api_key=candidate_keys[global_index])
        return ''

# request for three times
def get_completion(prompt, model, maxtry=5):
    response = ''
    try_number = 0
    while len(response) == 0:
        try_number += 1
        if try_number == maxtry: 
            print (f'fail for {maxtry} times')
            break
        response = func_get_completion(prompt, model)
    return response

# polish chatgpt's outputs
def func_postprocess_chatgpt(response):
    response = response.strip()
    if response.startswith("output"): response = response[len("output"):]
    if response.startswith("Output"): response = response[len("Output"):]
    response = response.strip()
    if response.startswith(":"):  response = response[len(":"):]
    response = response.strip()
    response = response.replace('\n', '')
    response = response.strip()
    return response


# ---------------------------------------------------------------------
## convert image/video into GPT4 support version
def func_image_to_base64(image_path, grey_flag=False): # support more types
    image = cv2.imread(image_path)
    if grey_flag:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return func_opencv_to_base64(image)

def func_opencv_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# deal with text
def func_nyp_to_text(npy_path):
    text = np.load(npy_path).tolist()
    text = text.strip()
    text = text.replace('\n', '') # remove \n
    text = text.replace('\t', '') # remove \t
    text = text.strip()
    return text

# support two types: (video) or (frames in dir)
def sample_frames_from_video(video_path, samplenum=3):
    if os.path.isdir(video_path): # already sampled video, frame store in video_path
        select_frames = sorted(glob.glob(video_path + '/*'))
        select_frames = select_frames[:samplenum]
        select_frames = [cv2.imread(item) for item in select_frames]
    else: # original video
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if ret == False: break
            frames.append(frame)
        cap.release()
        
        # return frames
        while len(frames) < samplenum:
            frames.append(frames[-1])
        
        tgt_length = int(len(frames)/samplenum)*samplenum
        frames = frames[:tgt_length]
        indices = np.arange(0, len(frames), int(len(frames) / samplenum)).astype(int).tolist()
        print ('sample indexes: ', indices)
        assert len(indices) == samplenum
        select_frames = [frames[index] for index in indices]
    assert len(select_frames) == samplenum, 'actual sampled frames is ont equal to tgt samplenum'
    return select_frames


# ---------------------------------------------------------------------
## Emotion api
# ---------------------------------------------------------------------
# 20 images per time, image_paths to load the images, candidate_list contains or the optional categories
def get_image_emotion_batch(image_paths, candidate_list, sleeptime=0, grey_flag=False, model='gpt-5-mini'):
    # Request GPT to output a JSON array. Each element must be an object with keys:
    #   name: original filename or an identifier
    #   probabilities: an object mapping each candidate label to a probability (float), sums to 1.0
    # Example single-item: {"name":"img.jpg","probabilities":{"angry":0.7,"happy":0.2,"relaxed":0.05,"sad":0.05}}
    # The model should return a JSON array containing one object per input image, in the same order.
    prompt = [
                {
                    "type":  "text", 
                    "text": (
                        f"Please play the role of a dog facial expression classification expert and recognize dog emotions from 4 category(angry/happy/relaxed/sad). We provide {len(image_paths)} images. "
                        "Focus primarily on facial expressions and demeanor, combined with body posture and movements. Think carefully on the typical elements of each expression before recognizing and try to get more accurate answers based on details. For each image, return the "
                        "probability for each of the provided categories so that the probabilities for each image sum to 1.0. "
                        f"Here are the optional categories: {candidate_list}.\n"
                        "IMPORTANT: Output MUST be valid JSON only (no extra commentary). The JSON should be an array, "
                        "each element an object with keys: \"name\" and \"probabilities\". \"probabilities\" must be an object mapping label->probability (float). "
                        "Example: [{\"name\":\"img1.jpg\", \"probabilities\": {\"angry\":0.7,\"happy\":0.2,\"relaxed\":0.05,\"sad\":0.05}}, ...]"
                    )
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path, grey_flag),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    
    # 调用API
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    
    # API调用后等待，避免速率限制（减少到5秒，新版API更稳定）
    if sleeptime > 0:
        print(f"⏳ 等待 {sleeptime} 秒...")
        time.sleep(sleeptime)
    
    return response

def get_evoke_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-5-mini'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a emotion recognition expert. We provide {len(image_paths)} images. \
                              Please recognize sentiments evoked by these images (i.e., guess how viewer might emotionally feel after seeing these images.) \
                              If there is a person in the image, ignore that person's identity. \
                              For each image, please sort the provided categories from high to low according to the similarity with the input image. \
                              Here are the optional categories: {candidate_list}. If there is a person in the image, ignore that person's identity. \
                              The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

def get_micro_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-5-mini'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a micro-expression recognition expert. We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
                              For each image, please sort the provided categories from high to low according to the similarity with the input image. \
                              The expression may not be obvious, please pay attention to the details of the face. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

# # 20 images per time
# def get_audio_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
#     prompt = [
#                 {
#                     "type":  "text", 
#                     "text": f"Please play the role of a audio expression classification expert. We provide {len(image_paths)} audios, each with an image of Mel spectrogram. \
#                               Please ignore the speaker's identity and recognize the speaker's expression from the provided Mel spectrogram. \
#                               For each sample, please sort the provided categories from high to low according to the top 5 similarity with the input. \
#                               Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each audio."
#                 }
#             ]
#     for ii, image_path in enumerate(image_paths):
#         prompt.append(
#             {
#                 "type":  f"audio-{ii+1}",
#                 "image": func_image_to_base64(image_path),
#             }
#         )
#     print (prompt[0]['text']) # debug
#     for item in prompt: print (item['type']) # debug
#     time.sleep(sleeptime)
#     response = get_completion(prompt, model)
#     response = func_postprocess_chatgpt(response)
#     print (response)
#     return response


def get_text_emotion_batch(npy_paths, candidate_list, sleeptime=0, model='gpt-5-mini'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a textual emotion classification expert. We provide {len(npy_paths)} texts. \
                              Please recognize the speaker's expression from the provided text. \
                              For each text, please sort the provided categories from high to low according to the top 5 similarity with the input. \
                              Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each text."
                }
            ]
    for ii, npy_path in enumerate(npy_paths):
        prompt.append(
            {
                "type":  f"text",
                "text": f"{func_nyp_to_text(npy_path)}",
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# 20 images per time
def get_video_emotion_batch(video_paths, candidate_list, sleeptime=0, samplenum=3, model='gpt-5-mini'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a video expression classification expert. We provide {len(video_paths)} videos, each with {samplenum} temporally uniformly sampled frames. Please ignore the speaker's identity and focus on their facial expression. \
                              For each video, please sort the provided categories from high to low according to the top 5 similarity with the input video. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each video."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        video_frames = sample_frames_from_video(video_path, samplenum)
        for jj, video_frame in enumerate(video_frames):
            prompt.append(
                    {
                        "type":  f"video{ii+1}_image{jj+1}",
                        "image": func_opencv_to_base64(video_frame),
                    },
            )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_multi_emotion_batch(video_paths, candidate_list, sleeptime=0, samplenum=3, model='gpt-5-mini'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a video expression classification expert. We provide {len(video_paths)} videos, each with the speaker's content and {samplenum} temporally uniformly sampled frames.\
                              Please ignore the speaker's identity and focus on their emotions. Please ignore the speaker's identity and focus on their emotions. \
                              For each video, please sort the provided categories from high to low according to the top 5 similarity with the input video. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on their emotions. The output format should be {{'name':, 'result':}} for each video."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        # convert video_path to text path
        split_paths = video_path.split('/')
        split_paths[-2] = 'text'
        split_paths[-1] = split_paths[-1].rsplit('.', 1)[0] + '.npy'
        text_path = "/".join(split_paths)
        assert os.path.exists(text_path)
        prompt.append(
                {
                    "type": "text",
                    "text": f"{func_nyp_to_text(text_path)}",
                },
        )

        # read frames
        video_frames = sample_frames_from_video(video_path, samplenum=3)
        for jj, video_frame in enumerate(video_frames):
            prompt.append(
                    {
                        "type":  f"video{ii+1}_image{jj+1}",
                        "image": func_opencv_to_base64(video_frame),
                    },
            )
       
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response
