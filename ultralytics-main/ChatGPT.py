# import openai
# # openai.api_key = "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"
# openai.api_key = "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"
#
# # 通过 `系统(system)` 角色给 `助手(assistant)` 角色赋予一个人设
# messages = [{'role': 'system', 'content': '你是一个乐于助人的诗人。'}]
# # # 在 messages 中加入 `用户(user)` 角色提出第 1 个问题
# # messages.append({'role': 'user', 'content': '作一首诗，要有风、要有肉，要有火锅、要有雾，要有美女、要有驴！'})
# # 调用接口
# response = openai.ChatCompletion.create(
#     model='gpt-3.5-turbo',
#     messages=messages,
# )
# print(response['choices'][0]['message']['content'])
# # # 在 messages 中加入 `助手(assistant)` 的回答
# # messages.append({
# #     'role': response['choices'][0]['message']['role'],
# #     'content': response['choices'][0]['message']['content'],
# # })
# # # 在 messages 中加入 `用户(user)` 角色提出第 2 个问题
# # messages.append({'role': 'user', 'content': '好诗！好诗！'})
# # # 调用接口
# # response = openai.ChatCompletion.create(
# #     model='gpt-3.5-turbo',
# #     messages=messages,
# # )
# # # 在 messages 中加入 `助手(assistant)` 的回答
# # messages.append({
# #     'role': response['choices'][0]['message']['role'],
# #     'content': response['choices'][0]['message']['content'],
# # })
# # 查看整个对话
#
#
#
#
# import requests
# openai_secret_key = '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'
# message ="Hello!"
# headers = {
#     'Content-Type': 'application/json',
#     'Authorization': f'Bearer {openai_secret_key}'
# }
# data = {
#     "model": "gpt-3.5-turbo",
#     "messages": [{"role": "user", "content": message}],
#     "temperature": 0.7
# }
# response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
# response_data = response.json()
# text = response_data["choices"][0]['message']['content']
# # print(text)

import torch
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = torch.nn.LayerNorm((5, embedding_dim))            # 在embedding_dim维度上做归一化

print(layer_norm.weight.shape)
print(layer_norm.bias.shape)

output = layer_norm(embedding)
print(output.shape)

# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)

layer_norm = torch.nn.LayerNorm([C, H, W])                # 在每一个特征层维度上做归一化
output = layer_norm(input)
print(layer_norm.weight.shape)
print(layer_norm.bias.shape)
print(output.shape)


