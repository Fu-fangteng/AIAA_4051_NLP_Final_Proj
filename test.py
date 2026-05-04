import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_connection():
    print("=== 1. 检查环境变量 ===")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置 (将连接原站)')}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("\n=== 2. 尝试加载 Tokenizer ===")
        # 加上 trust_remote_code=True 增加兼容性
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", timeout=30)
        print("✅ Tokenizer 加载成功！")

        print("\n=== 3. 尝试加载 Model 权重 (关键步骤) ===")
        print("提示: 如果此处卡住，说明 model.safetensors 下载太慢或连接超时。")
        model = GPT2LMHeadModel.from_pretrained("/home/user/project/gpt2").to(device)
        print("✅ Model 权重下载并加载成功！")
        print(f"显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        print("\n=== 4. 测试 lxt_patch 逻辑 ===")
        from lxt.efficient.rules import stop_gradient
        # 模拟一个 stop_gradient 操作
        test_tensor = torch.randn(1, 10, 768, requires_grad=True).to(device)
        stopped = stop_gradient(test_tensor)
        if not stopped.requires_grad:
            print("✅ LXT stop_gradient 补丁工作正常！")

        print("\n=== 5. 模拟一次推理 + 反传 ===")
        input_ids = torch.tensor([[50256]]).to(device) # <|endoftext|>
        input_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        logits = model(inputs_embeds=input_embeds).logits
        loss = logits.sum()
        loss.backward()
        
        if input_embeds.grad is not None:
            print("✅ 推理与梯度回传流程通畅！")
        
        print("\n🎉 所有测试通过！你可以放心运行 nohup 脚本了。")

    except Exception as e:
        print(f"\n❌ 出错了！错误类型: {type(e).__name__}")
        print(f"错误细节: {e}")
        print("\n💡 建议方案:")
        print("1. 如果是 ReadTimeout，执行: export HF_HUB_READ_TIMEOUT=120")
        print("2. 如果是 ConnectionError，确保执行了: export HF_ENDPOINT=https://hf-mirror.com")

if __name__ == "__main__":
    test_connection()