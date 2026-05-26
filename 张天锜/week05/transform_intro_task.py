import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#实例文本
text = "杀人放火厉飞雨，救死扶伤寒天尊！"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#将文本编码为token
input_ids = tokenizer.encode(text, return_tensors='pt')

model = GPT2LMHeadModel.from_pretrained('gpt2')

#定义优化器
optimizer = torch.optim.Adam(model.parameters(),lr= 5e-5)

#循环训练
for epoch in range(5):
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'Epoch{epoch + 1}, loss: {loss.item()}')
    
#进行文本生成
generated = model.generate(input_ids, max_length=50, num_return_sequence=1)
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print("Generated Text", generated_text)

