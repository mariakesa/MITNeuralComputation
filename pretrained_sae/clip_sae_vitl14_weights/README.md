---
license: mit
datasets:
- zer0int/CLIP-adversarial-typographic-attack_text-image
- SPRIGHT-T2I/spright_coco
base_model:
- openai/clip-vit-large-patch14
pipeline_tag: zero-shot-image-classification
library_name: transformers
---
### CLIP ViT-L/14 finetune: SAE-informed adversarial training

- SAE = Sparse autoencoder
- Accuracy ImageNet/ObjectNet [my GmP](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14): 91% > SAE (this): 89% > OpenAI pre-trained: 84.5%
- But, it's fun to use with e.g. Flux.1 - get the [Text-Encoder TE only version](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14/resolve/main/ViT-L-14-GmP-SAE-TE-only.safetensors?download=true) ⬇️ and try it!
- And this SAE CLIP has best results for linear probe @ [LAION-AI/CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) (see below)

- This CLIP [direct download](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14/resolve/main/ViT-L-14-GmP-SAE-TE-only.safetensors?download=true) is also the best CLIP to use for [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).
- Required: Use with my [zer0int/ComfyUI-HunyuanVideo-Nyan](https://github.com/zer0int/ComfyUI-HunyuanVideo-Nyan) node (changes influence of LLM vs. CLIP; otherwise, difference is very little).


<video controls autoplay src="https://cdn-uploads.huggingface.co/production/uploads/6490359a877fc29cb1b09451/g0vO1N4JalPp8oIAq5v38.mp4"></video>


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6490359a877fc29cb1b09451/m6Qty30oeS7A8cDYvLWme.png)

- Interesting things with adversarial robustness to try: Right-click and download individual images: [Image 1](https://raw.githubusercontent.com/zer0int/CLIP-SAE-finetune/refs/heads/CLIP-vision/bwcat_cat.png) -- [Image 2](https://raw.githubusercontent.com/zer0int/CLIP-SAE-finetune/refs/heads/CLIP-vision/bwcat_dog.png) -- [Image 3](https://raw.githubusercontent.com/zer0int/CLIP-SAE-finetune/refs/heads/CLIP-vision/bwcat_notext.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6490359a877fc29cb1b09451/CN7xMe5ZPfLVWST-RF6Qn.png)
- Upload each into zero-shot [hopefully available soon on the right here->]
- Try labels (class names): a photo of a cat, a photo of a dog, a photo of a text
- Repeat the same with e.g. [my GmP models](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) models and see what happens. =)
- I'm really hoping the HF format .safetensors conversion didn't mess anything up (it happens!); just in case it did, or if there's no inference API available to use:
- I put a script that will do the same thing (on the not-converted model) on my GitHub repo. Plus, you can just reproduce the fine-tune yourself, as that code is also available! 🤗
- 👉 All training info & code: [github.com/zer0int/CLIP-SAE-finetune](https://github.com/zer0int/CLIP-SAE-finetune)
- ☕ [Buy me a coffee](https://ko-fi.com/zer0int)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6490359a877fc29cb1b09451/_Bp8DoxgkOjhau5EnShtW.png)

