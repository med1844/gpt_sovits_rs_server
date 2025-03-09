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

## Compiling on Windows

### Prerequisites

- `rustup` and `cargo`
  - Note that you might have to install MSVC to proceed

After you cloned this repository, create a python venv so that we can download libtorch.

### Prepare libtorch

```powershell
uv venv
.\.venv\Scripts\activate
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Then setup environment vars for `cargo` to compile and run:

```powershell
$env:LIBTORCH_USE_PYTORCH = 1
$env:LIBTORCH = ".\.venv\Lib\site-packages\torch\lib\"
```

### Compile

```powershell
cargo build -r
```

You can find the compiled exe in `target/release`.

### Standalone exe

It's much simpler to make distributable packages with compiled exe, with all dependencies included.

Here's how the files would look like:

```bash
> tree .
.
├── 250113_0059_processed_3.wav
├── asmjit.dll
├── c10.dll
├── c10_cuda.dll
├── config.json
├── cudnn64_9.dll
├── cudnn_engines_precompiled64_9.dll
├── cudnn_engines_runtime_compiled64_9.dll
├── cudnn_graph64_9.dll
├── cudnn_heuristic64_9.dll
├── fbgemm.dll
├── models
│   ├── bert_model.pt
│   ├── gpt_sovits_model.pt
│   └── ssl_model.pt
├── nvToolsExt64_1.dll
├── resource
│   ├── en_word_dict.json
│   ├── g2pw
│   │   ├── CHARS.txt
│   │   ├── LABELS.txt
│   │   ├── MONOPHONIC_CHARS.txt
│   │   ├── POLYPHONIC_CHARS.txt
│   │   ├── bert-base-chinese_s2t_dict.txt
│   │   ├── bopomofo_to_pinyin_wo_tune_dict.json
│   │   ├── char_bopomofo_dict.json
│   │   ├── dict_mono_chars.json
│   │   ├── dict_poly_chars.json
│   │   ├── dict_poly_index_list.json
│   │   └── dict_poly_index_map.json
│   ├── g2pw.pt
│   ├── g2pw_tokenizer.json
│   ├── mini-bart-g2p.pt
│   ├── rule.pest
│   ├── symbols_v2.json
│   ├── tokenizer.mini-bart-g2p.json
│   └── zh_word_dict.json
├── server.exe
├── torch_cpu.dll
├── torch_cuda.dll
└── uv.dll

3 directories, 38 files

projects/gpt_sovits_rs_win/standalone_server on aws (us-east-1)
> du -sh .
4.6G    .
```

