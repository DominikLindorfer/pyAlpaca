<p align="center">
  <img src="https://user-images.githubusercontent.com/21077042/234803304-d01984eb-4cf0-4f1f-ae13-4ba285c09ce7.png" width="350">
</p>


# pyAlpaca: Instruction-following LLaMA Model Trained to Output Python-Code from General Instructions
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project aims to build and share a low-resource instruction-following LLaMA model for python code generation and is based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) as well as [Code Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [llama.cpp](https://github.com/tatsu-lab/stanford_alpaca). While the former projects aim towards general purpose LLMs, this project aims to build a code-assistant that specializes in generating code in a singular language - Python.

## Non-sped up Demo of pyAlpaca-13B using [llama.cpp](https://github.com/ggerganov/llama.cpp) Inference on an Intel i-13600K with 32GB RAM

**Prompt:** "Write a Python program that fits a spline curve into data."

https://user-images.githubusercontent.com/21077042/234803452-2b5cb113-58d4-48d6-aad5-9c03f13b8d8b.mp4


## Additional Simplistic Usage with [llama_cpp Python-Bindings]( https://github.com/abetlen/llama-cpp-python )

    from llama_cpp import Llama
    llm = Llama(model_path="./llama.cpp/models/ggml-model-f16.bin")
    output = llm("### Instruction: Write a Python program to perform insertion sort on a list. ### Response:", max_tokens=256, stop=["Output"], echo=True)
    print(output)

## Model Weights on HuggingFace Repo

Models are shared on HuggingFace Repos here. (Uploading...)

## Examples:

**Prompt:** "Write a Python program that fits a spline curve into data."

**Output:**

    import numpy as np
    from scipy.optimize import curve_fit

    def func(x, a, b, c):
        return a*x**2 + b*x + c

    x_data = np.array([1, 2, 3, 4, 5])
    y_data = func(x_data, 2, 3, 5)

    popt, pcov = curve_fit(func, x_data, y_data)

    print("Best-fit parameters:", popt)
    print("Parameter uncertainties:", pcov)

**Prompt:** "Write a Python program using binary search."

**Output:**

    def binary_search(arr, target): 
        left = 0
        right = len(arr)-1

        while left<=right: 

            mid = (left+right)//2

            # Check if the element is present at the middle itself 
            if arr[mid] == target: 
                return mid 

            # If the element is smaller than mid, it can only  
            # be present in left subarray 
            elif arr[mid] > target: 
                right = mid - 1 

            # Else the element can only be present in right  
            # subarray 
            else: 
                left = mid + 1
        return -1


**Prompt:** "Write a Python function to delete a number from an array."

**Output:**

    def delete_number(arr, num):
        if num in arr:
            arr.remove(num)
        return arr

    arr = [1, 2, 3, 4, 5]
    num = 3

    print(delete_number(arr, num)) # Output: [1, 2, 4, 5]


**Prompt:** "Write a Python function to perform insertion sort on a list."

**Output:**

    def insertion_sort(list):
        for i in range(1, len(list)):
            key = list[i]
            j = i-1
            while j >= 0 and list[j] > key:
                list[j+1] = list[j]
                j -= 1
            list[j+1] = key
        return list

**Prompt:** Describe quantization of datatypes in machine learning.

**Output:**

    ### Quantization of data

    Quantization is the process of reducing the amount of data needed to represent a signal or image. It involves mapping the original data onto a lower-dimensional space while retaining the essential information. In machine learning, this can be used to reduce the amount of data needed for training, or to reduce the complexity of the model.

    ### Quantization of datatypes

    In machine learning, datatypes are often quantized to reduce the amount of data needed for training, or to reduce the complexity of the model. For example, images can be represented using fewer colors or less resolution, and audio can be represented using fewer frequencies or less time resolution. Additionally, integers can be represented using fewer bits, and floats can be represented using fewer decimal places.

    ### Benefits of quantization

    Quantization can reduce the amount of data needed for training, making it easier to store and process data. Additionally, it can reduce the complexity of models, making them faster and more efficient. Finally, it can also reduce the amount of computation needed, making it easier to deploy models on limited hardware systems. 

    ### Drawbacks of quantization

    The main drawback of quantization is the loss of information. By reducing the amount of data, we are also reducing the amount of information that can be stored or processed. This can lead to reduced accuracy, especially in tasks such as image or speech recognition. Additionally, some models may not be able to handle the reduced data, and may require more data for training. Finally, there may be some cases where the reduced accuracy is not acceptable, and the full amount of data may be needed. 


## Training using Deepspeed

pyAlpaca has been trained on **1(!) A100 40G GPU as well as 256GB RAM**, which is commonly found in older research clusters.
The original Stanford Alpaca model has been trained on 8 A100 80G GPUs in FSDP `full_shard` mode - a configuration not available to many researchers, enthusiasts or me. Thus, this project relies heavily on ([Microsoft's Deepspeed Library](www.deepspeed.ai)) which not only reduces the GPU resources needed but can offload to RAM using the Deepspeed Stage 3 approach. Please check out their papers in Ref [2,3 & 4]. 

The deepspeed configuration that was used is:
    
    ds_config.json:
    
    {
    "zero_optimization": {
            "stage": 3,
            "contiguous_gradients": true,
            "stage3_max_live_parameters": 0,
            "stage3_max_reuse_distance": 0,
            "stage3_prefetch_bucket_size": 0,
            "stage3_param_persistence_threshold": 100,
            "reduce_bucket_size": 100,
            "sub_group_size": 100000000,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            },
            "stage3_gather_16bit_weights_on_model_save": true
         },    
    "optimizer": {
            "type": "Adam",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
         },
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "initial_scale_power": 32,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": false
    }


## Data Release
[`data/pyinstructions.json`](./data/pyinstructions.json) contains ~9.5K instruction-following data relevant to Python used for fine-tuning the pyAlpaca model, following the proposed method in [5].
This JSON file is a list of dictionaries, each dictionary contains the following fields:
- `instruction`: `str`, describes the task the model should perform.
- `input`: `str`, optional context or input for the task.
- `output`: `str`, the answer to the instruction as generated by `text-davinci-003`.

## Fine-tuning

The pyAlpaca model is fine-tuned using HuggingFace's Trainer an the following parameters:

* Batch size: 128
* Learning rate: 2e-5
* Epochs: 3 (7B and 13B) and 5 (7B-5)
* Max length: 512
* Weight decay: 0

Below is the used command that fine-tunes LLaMA models with our dataset on a machine with 1 A100 40G GPU using deepspeed, as described above.
Replace `<your_path_to_hf_converted_llama_ckpt_and_tokenizer>` with the path to your HuggingFace converted checkpoint and tokenizer and `<your_output_dir>` with the directory to store the output.

```bash
torchrun train.py \
    --model_name_or_path ./cpt_HF_13B/ \
    --data_path "<your_path_to_hf_converted_llama_ckpt_and_tokenizer>" \
    --fp16 True \
    --output_dir "<your_output_dir>" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config_stage3.json \
    --tf32 False
```

### On-Going and Future Steps

* 8-Bit and 4-Bit Quantization
* pyAlpaca-30B Model
* Training and Comparison of LoRA Approach
* Comparison of LLaMA Adapter in 7B and 13B
* Extending the Dataset using modified Leetcode-10K
* Extending the Dataset using modified StackOverflow BigQuery-DS

### Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{alpaca,
  author = {Dominik Lindorfer},
  title = {pyAlpaca: Instruction-following LLaMA Model Trained to Output Python-Code from General Instructions},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dominiklindorfer/pyAlpaca}},
}
```

Please also cite the original LLaMA paper [1] and the Self-Instruct paper [5], as well as Microsoft's Deepspeed Library [2,3,4].

### References

[1]: LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1

[2]: ZeRO-Offload: Democratizing Billion-Scale Model Training. Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He. https://arxiv.org/abs/2101.06840

[3]: ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. https://arxiv.org/abs/2104.07857

[4]: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. https://arxiv.org/abs/1910.02054

[5]: Self-Instruct: Aligning Language Model with Self Generated Instructions. Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. https://arxiv.org/abs/2212.10560
