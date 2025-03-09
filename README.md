## How to run

### Prepare models and resources

Refer to [`gpt_sovits_rs`](https://github.com/second-state/gpt_sovits_rs?tab=readme-ov-file#exporting-gpt-sovits-training-results) instructions to output models. Put them in `models` folder.

```
> tree models/
models/
├── bert_model.pt
├── gpt_sovits_model.pt
└── ssl_model.pt
```

In the same section, download `g2pw.pt` and put it in `resource`:

```
> tree resource/ -L 1
resource/
├── en_word_dict.json
├── g2pw
├── g2pw.pt
├── g2pw_tokenizer.json
├── mini-bart-g2p.pt
├── rule.pest
├── symbols_v2.json
├── tokenizer.mini-bart-g2p.json
└── zh_word_dict.json

1 directory, 8 files
```

### Prepare reference audio & config

```
> cat config.json
{
  "level_filter": "info",
  "ref_path": "250113_0059_processed_3.wav",
  "ref_text": "她被人施了诅咒，只要诅咒不解，便会一直昏睡下去。",
  "spk_name": "c_sf4_normal",
  "host": "127.0.0.1",
  "port": 3000
}
```

Replace `ref_path` and `ref_text`.  It's optional to change `spk_name`.

### Run

```
LIBTORCH=/usr/lib/libtorch LD_LIBRARY_PATH=/usr/lib/libtorch/lib cargo run -r
```
