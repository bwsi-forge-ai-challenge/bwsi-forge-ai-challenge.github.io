"""Real backends built on small open-weight Hugging Face models. These are the *base* models: each one
accepts `checkpoint:` (a LoRA folder from your fine-tuning) and `style_prompt:` (for the prompted baseline
in E1), so the same backend serves zero-shot, prompted and fine-tuned runs.

    blip       C  Salesforce/blip-image-captioning-base   ~1 GB
    musicgen   M  facebook/musicgen-small                  ~2 GB, non-commercial weights
    clap_tags  L  laion/clap-htsat-unfused                 ~0.6 GB, zero-shot tags filled into a template
    sd_turbo   G  stabilityai/sd-turbo                     ~2.5 GB, 1 denoising step, non-commercial weights

clap_tags is a placeholder for L, not something you can fine-tune as-is: it is the "tags filled into
templates" approach from the Music Captioning lesson. Replace it with a trained captioner (LP-MusicCaps,
Qwen2-Audio, or a frozen CLAP/MERT encoder plus a small decoder) by writing a new backend.
"""

import torch

from .. import media
from ..stage import Stage, pick_device, register


def _features(out):
    """`get_*_features` returns a tensor in older transformers and an output object in newer ones."""
    return out if torch.is_tensor(out) else out.pooler_output


def _with_lora(model, checkpoint):
    if not checkpoint:
        return model
    from peft import PeftModel
    return PeftModel.from_pretrained(model, checkpoint)


def _join(text, style_prompt):
    return f"{text}, {style_prompt}" if style_prompt else text


@register("blip")
class BlipCaptioner(Stage):
    kinds = ("C",)

    def load(self):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        name = self.model_name or "Salesforce/blip-image-captioning-base"
        self.device = pick_device(self.device_request)
        self.processor = BlipProcessor.from_pretrained(name)
        model = BlipForConditionalGeneration.from_pretrained(name)
        self.model = _with_lora(model, self.checkpoint).to(self.device).eval()

    @torch.no_grad()
    def generate(self, image, seed):
        # BLIP continues the style prompt as a prefix, e.g. "a grainy noir photograph of"
        prompt = self.style_prompt or None
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        ids = self.model.generate(**inputs, max_new_tokens=int(self.params.get("max_new_tokens", 40)),
                                  num_beams=int(self.params.get("num_beams", 3)))
        return self.processor.decode(ids[0], skip_special_tokens=True)


@register("musicgen")
class MusicGenComposer(Stage):
    kinds = ("M",)

    def load(self):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        name = self.model_name or "facebook/musicgen-small"
        self.device = pick_device(self.device_request)
        self.processor = AutoProcessor.from_pretrained(name)
        model = MusicgenForConditionalGeneration.from_pretrained(name)
        self.model = _with_lora(model, self.checkpoint).to(self.device).eval()
        self.sample_rate = model.config.audio_encoder.sampling_rate
        self.frame_rate = model.config.audio_encoder.frame_rate

    @torch.no_grad()
    def generate(self, text, seed):
        seconds = float(self.params.get("duration", media.MIN_SECONDS))
        inputs = self.processor(text=[_join(text, self.style_prompt)], padding=True, return_tensors="pt").to(self.device)
        audio = self.model.generate(**inputs, do_sample=True, guidance_scale=float(self.params.get("guidance_scale", 3.0)),
                                    max_new_tokens=int(seconds * self.frame_rate))
        return media.Audio(audio[0, 0].float().cpu().numpy(), self.sample_rate)


@register("clap_tags")
class ClapTagListener(Stage):
    kinds = ("L",)

    VOCAB = {
        "genre": ["jazz", "rock", "classical", "electronic", "folk", "hip-hop", "ambient", "pop", "chiptune"],
        "mood": ["happy", "sad", "calm", "tense", "energetic", "dreamy", "dark"],
        "instrument": ["piano", "acoustic guitar", "electric guitar", "drums", "strings", "synthesizer",
                       "trumpet", "flute", "bass"],
    }
    TEMPLATE = "A {mood} {genre} piece with {instruments}, around {tempo} BPM."

    def load(self):
        from transformers import ClapModel, ClapProcessor
        name = self.model_name or "laion/clap-htsat-unfused"
        self.device = pick_device(self.device_request)
        self.processor = ClapProcessor.from_pretrained(name)
        self.model = ClapModel.from_pretrained(name).to(self.device).eval()
        # Your style's vocabulary and sentence shape can be set in the config: vocab: {mood: [...]}, template: "..."
        self.vocab = {**self.VOCAB, **self.params.get("vocab", {})}
        self.template = self.params.get("template", self.TEMPLATE)
        self.word_embeds = {}
        with torch.no_grad():
            for slot, words in self.vocab.items():
                tok = self.processor.tokenizer([f"{w} music" for w in words], padding=True, return_tensors="pt")
                emb = _features(self.model.get_text_features(**tok.to(self.device)))
                self.word_embeds[slot] = torch.nn.functional.normalize(emb, dim=-1)

    @torch.no_grad()
    def embed_audio(self, audio):
        x = media.resample(audio.samples, audio.sample_rate, 48_000)
        feats = self.processor.feature_extractor(x, sampling_rate=48_000, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(_features(self.model.get_audio_features(**feats)), dim=-1)

    def generate(self, audio, seed):
        a = self.embed_audio(audio)
        best = {slot: (a @ e.T)[0] for slot, e in self.word_embeds.items()}
        top = lambda slot, k: [self.vocab[slot][i] for i in best[slot].topk(k).indices.tolist()]
        caption = self.template.format(mood=top("mood", 1)[0], genre=top("genre", 1)[0],
                                       instruments=" and ".join(top("instrument", 2)), tempo=media.estimate_bpm(audio))
        return _join(caption, self.style_prompt)


@register("sd_turbo")
class SDTurboGenerator(Stage):
    kinds = ("G",)

    def load(self):
        from diffusers import AutoPipelineForText2Image
        name = self.model_name or "stabilityai/sd-turbo"
        self.device = pick_device(self.device_request)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = AutoPipelineForText2Image.from_pretrained(name, torch_dtype=dtype, variant="fp16").to(self.device)
        if self.checkpoint:
            self.pipe.load_lora_weights(self.checkpoint)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, text, seed):
        generator = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(_join(text, self.style_prompt), num_inference_steps=int(self.params.get("steps", 1)),
                         guidance_scale=float(self.params.get("guidance_scale", 0.0)),
                         height=media.IMAGE_SIZE, width=media.IMAGE_SIZE, generator=generator).images[0]
