"""All user-facing text lives here, in one table per language (``en`` and ``ko`` keys must match)."""

from __future__ import annotations

import os
from typing import Any, Mapping

LANGS = ("en", "ko")
ENV_VAR = "SPECPAIR_LANG"

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "word.yes": "yes",
        "word.no": "no",
        "word.from_file": "from file",
        "word.implicit": "implicit default",
        "word.target": "target",
        "word.draft": "draft",
        "mock.banner": "PRACTICE MOCK - replays timings recorded with llama.cpp b12360 on CPU; your hardware will differ",
        "sum.tokens": "{n} tokens",
        "sum.special": "{name} id {id}, add: {add} ({src})",
        "header": "specpair {cmd} | rule set {runtime} (source: {url}, checked {date})",
        "check.target": "target: {path}",
        "check.draft": "draft : {path}",
        "check.verdict_ok": "verdict: compatible (token strings identical on {pct}% of ids)",
        "check.verdict_bad": "verdict: rejected ({n} rules failed)",
        "check.verdict_bad_one": "verdict: rejected (1 rule failed)",
        "check.notes": "notes:",
        "check.run": "Start llama-server with:",
        "check.run_warn": "Note: --spec-type defaults to none, so -md alone loads the draft but never uses it. "
                          "Options such as --draft-max were removed; use --spec-draft-n-max.",
        "check.rejected_hint": "llama-server would log 'the target and draft vocabs are not compatible' and keep "
                               "serving without the draft.",
        "check.hf_hint": "Compatible by tokenizer files. Convert both models to GGUF and check the GGUF files "
                         "before building a command.",
        "fail.vocab_type": "vocabulary types differ: target {target_type} vs draft {draft_type} "
                           "(tokenizer.ggml.model {target_model} vs {draft_model}, pre {target_pre} vs {draft_pre})",
        "fail.bos": "BOS handling differs (add_bos must match, and the BOS id must match while it is added): "
                    "add_bos {target_add} vs {draft_add}, BOS id {target_id} vs {draft_id}",
        "fail.eos": "EOS handling differs (add_eos must match, and the EOS id must match while it is added): "
                    "add_eos {target_add} vs {draft_add}, EOS id {target_id} vs {draft_id}",
        "fail.vocab_size": "vocabulary sizes differ by {diff} (limit {max}): target {target} vs draft {draft} tokens",
        "fail.token_text": "{count} token strings differ from id {start_id} on; first mismatch at id {first_id}: "
                           "target {target_token} vs draft {draft_token} (first ones: {examples_text})",
        "short.vocab_type": "{target_type} vs {draft_type}",
        "short.bos": "add_bos {target_add} vs {draft_add}, id {target_id} vs {draft_id}",
        "short.eos": "add_eos {target_add} vs {draft_add}, id {target_id} vs {draft_id}",
        "short.vocab_size": "size {target} vs {draft} (diff {diff} > {max})",
        "short.token_text": "{count} strings differ, first at id {first_id}",
        "note.implicit_bos": "{side}: no add_bos_token key in the file, so llama.cpp uses its default ({value}) for "
                             "{vocab_type} vocabularies with pre '{pre}'",
        "note.implicit_eos": "{side}: no add_eos_token key in the file, so llama.cpp uses its default ({value}) for "
                             "{vocab_type} vocabularies with pre '{pre}'",
        "note.bos_id_differs": "BOS ids differ ({target_id} vs {draft_id}) but add_bos is off on both sides, "
                               "so llama.cpp does not reject the pair",
        "note.eos_id_differs": "EOS ids differ ({target_id} vs {draft_id}) but add_eos is off on both sides, "
                               "so llama.cpp does not reject the pair",
        "note.size_boundary": "vocabulary sizes differ by exactly {diff}, the limit ({max}): allowed, but one more "
                              "token would be rejected",
        "note.size_diff_allowed": "vocabulary sizes differ by {diff} (limit {max}); ids from {unchecked_from} on "
                                  "are not compared",
        "note.misleading_vocab_type": "both vocabularies are {vocab_type}; llama-server may still log 'draft model "
                                      "vocab type must match' - the real reason is one of the failed rules above",
        "note.template_differs": "chat templates differ; llama-server formats prompts with the target's template only",
        "note.gguf_padding": "the {longer} vocabulary has {extra} extra [PAD] slots; GGUF conversion pads the "
                             "vocabulary to the embedding size, so sizes can differ from the Hugging Face files",
        "note.hf_input": "a Hugging Face tokenizer folder was used; GGUF conversion can change the vocabulary size "
                         "(padding) and the add_bos default, so confirm with the GGUF files",
        "note.dedicated_head": "{side} architecture '{arch}' looks like a dedicated draft head (EAGLE/MTP style) "
                               "that this rule set does not judge",
        "scan.target": "target: {path}",
        "scan.folder": "folder: {path}",
        "scan.empty": "no .gguf files found (looked in the folder and its direct sub-folders)",
        "scan.col.file": "file",
        "scan.col.status": "status",
        "scan.col.detail": "detail",
        "scan.same_ids": "token strings identical on {pct}% of ids",
        "scan.summary": "{ok} of {total} files can be used as a draft",
        "bench.title": "specpair bench | {n} servers, {prompts} prompts, n_predict {n_predict}, repeat {repeat}",
        "bench.col.config": "config",
        "bench.col.tokens": "tokens",
        "bench.col.tps": "t/s",
        "bench.col.drafted": "drafted",
        "bench.col.accepted": "accepted",
        "bench.col.acceptance": "acceptance",
        "bench.col.speedup": "vs baseline",
        "bench.col.prompt": "prompt",
        "bench.baseline": "baseline",
        "bench.per_prompt": "per prompt:",
        "bench.warnings": "warnings:",
        "bench.no_warnings": "no warnings: every draft was used and none is slower than baseline",
        "bench.noise": "Speeds come from the servers' own timings. Compare runs on the same machine under the same "
                       "load and use --repeat to smooth noise.",
        "warn.draft_unused": "{label}: draft not used - the server reported no draft_n. The draft was rejected at "
                             "load time or --spec-type is none (llama-server keeps serving without speculation).",
        "warn.slower": "{label}: slower than baseline ({percent}%). A good acceptance rate does not pay off here; "
                       "try a smaller --spec-draft-n-max or a cheaper draft.",
        "warn.output_differs": "{label}: output differs from baseline (greedy decoding should give identical text); "
                               "check that both servers use the same target model and settings.",
        "demo.intro": "Synthetic GGUF headers (a few KB each) are checked by the real rules; the bench part replays "
                      "recorded timings through a local HTTP server. No model, GPU or network is used.",
        "demo.step1": "== 1/4 check {name}: every failing rule at once ==",
        "demo.step2": "== 2/4 check {name}: a pair llama.cpp accepts ==",
        "demo.step3": "== 3/4 scan a folder (a corrupt file does not stop the scan) ==",
        "demo.step4": "== 4/4 bench: draft not used, and a setting that is slower than baseline ==",
        "demo.outro": "Next: specpair demo --extract demo-files, then run the commands above on your own copies.",
        "demo.extracted": "Wrote {n} synthetic files to {dir}. Try:",
        "cli.error_prefix": "error",
        "cli.interrupted": "interrupted",
        "err.generic": "unexpected error",
        "err.not_found": "'{path}' does not exist. Check the path; pass a .gguf file or a folder with tokenizer.json.",
        "err.unreadable": "cannot read '{path}': {reason}",
        "err.not_a_folder": "'{path}' is not a folder. Pass the folder that holds your .gguf files.",
        "err.no_tokenizer_json": "folder '{path}' has no tokenizer.json. Pass a .gguf file or a Hugging Face "
                                 "tokenizer folder.",
        "err.not_gguf": "'{path}' is not a GGUF file (the 'GGUF' magic bytes are missing). Download it again or pass "
                        "the right file.",
        "err.gguf_version": "'{path}' uses GGUF version {version}; only versions 2 and 3 are supported.",
        "err.gguf_value_type": "'{path}' is corrupt: unsupported value type {type_id} for key '{key}' at byte {offset}.",
        "err.file_empty": "'{path}' is empty (0 bytes). Download it again or pass the right file.",
        "err.header_cut_fixed": "'{path}' ends inside the fixed GGUF header at byte {offset}. The file is cut off; "
                                "download it again.",
        "err.header_truncated": "the GGUF header of '{path}' ends in the middle of key '{key}' at byte {offset}. The "
                                "file is cut off; download it again (a partial file works only if the whole header "
                                "is present).",
        "err.no_tokenizer": "'{path}' has no tokenizer metadata ('{key}' is missing).",
        "err.no_tokenizer_split": "'{path}' is a later part of a split model and has no tokenizer keys. Use the "
                                  "first part '{first}' instead.",
        "err.tok_json_truncated": "'{path}' is truncated ({size} bytes; the JSON ends early). Download it again.",
        "err.tok_json_utf8": "'{path}' is not valid UTF-8 (first bad byte at offset {offset}). Download it again; "
                             "do not re-save it in another encoding.",
        "err.tok_json_invalid": "'{path}' is not valid JSON (line {line}, column {col}).",
        "err.tok_json_key": "'{path}' lacks the required key '{key}'; it does not look like a tokenizer.json.",
        "err.tok_json_invalid_ids": "'{path}' has token ids out of range (vocabulary of {n} slots).",
        "err.runtime_unsupported": "unsupported runtime '{runtime}'. Supported: {supported}.",
        "err.bench_connect": "cannot reach llama-server at {url}: {reason}. Start it first "
                             "(llama-server -m MODEL --port N) and check the URL.",
        "err.bench_timeout": "{url} did not answer within {timeout} seconds. Increase --timeout; a first request or "
                             "a CPU-only run can be slow.",
        "err.bench_health": "{url}/health answered HTTP {status}. A llama-server that is still loading answers 503; "
                            "wait until it reports 200 and retry.",
        "err.bench_http": "{url}/completion answered HTTP {status}.",
        "err.bench_no_timings": "{url} answered without timings.predicted_n / predicted_ms. Use a llama-server whose "
                                "/completion reports timings.",
        "err.bad_url": "'{url}' is not an http(s) URL, for example http://127.0.0.1:8080.",
        "err.bad_numbers": "--n-predict, --repeat and --timeout must be positive numbers.",
        "err.duplicate_label": "server labels must be unique; give each --spec its own LABEL=URL.",
        "err.prompts_unreadable": "cannot read prompts file '{path}': {reason}",
        "err.prompts_utf8": "prompts file '{path}' is not valid UTF-8 (first bad byte at offset {offset}).",
        "err.prompts_json": "prompts file '{path}': line {line} is not a JSON string or an object with a 'prompt'.",
        "err.prompts_empty": "prompts file '{path}' has no prompts.",
        "err.mock_conflict": "--mock cannot be combined with --base, --spec or --prompts.",
        "err.bench_args": "bench needs --base URL and at least one --spec [LABEL=]URL (or --mock).",
        "err.lang_unsupported": "unsupported language '{value}'; use en or ko.",
        "help.description": "Check whether a draft model can speculate for a target model (GGUF headers or Hugging "
                            "Face tokenizers) and measure the real speedup on llama-server.",
        "help.epilog": "exit codes: 0 ok, 1 verdict failed (rejected / no compatible draft / draft unused or slower), "
                       "2 bad input or connection error. Language: --lang, then SPECPAIR_LANG, then LC_ALL / "
                       "LC_MESSAGES / LANG.",
        "help.lang": "output language: en or ko (default: SPECPAIR_LANG, then the locale)",
        "help.version": "show the version and exit",
        "help.help": "show this help message and exit",
        "help.usage": "usage:",
        "help.positional": "positional arguments",
        "help.options": "options",
        "help.commands": "commands",
        "help.check": "check whether a draft model fits a target model",
        "help.check.target": "target model: .gguf file (first part if split) or Hugging Face tokenizer folder",
        "help.check.draft": "draft model: .gguf file or Hugging Face tokenizer folder",
        "help.runtime": "rule set to apply (default: llama.cpp@b12360)",
        "help.json": "print machine-readable JSON",
        "help.scan": "list which GGUF files in a folder can draft for a target",
        "help.scan.target": "target model (.gguf or tokenizer folder)",
        "help.scan.dir": "folder with .gguf files (its direct sub-folders are scanned too)",
        "help.bench": "compare running llama-servers: tokens/s, acceptance rate, speedup",
        "help.bench.base": "URL of the baseline server (no draft), e.g. http://127.0.0.1:8080",
        "help.bench.spec": "URL of a draft-enabled server, optionally LABEL=URL; repeat for more",
        "help.bench.prompts": "prompts file: one prompt per line, or .jsonl",
        "help.bench.n_predict": "tokens to generate per prompt (default 128)",
        "help.bench.repeat": "run each prompt this many times (default 1)",
        "help.bench.timeout": "seconds to wait for each server answer (default 120)",
        "help.bench.mock": "practice mock: replay recorded timings instead of real servers",
        "help.demo": "offline tour with synthetic files (practice mock)",
        "help.demo.extract": "copy the synthetic demo files into DIR instead of running the tour",
    },
    "ko": {
        "word.yes": "예",
        "word.no": "아니오",
        "word.from_file": "파일 값",
        "word.implicit": "암묵 기본값",
        "word.target": "대상",
        "word.draft": "드래프트",
        "mock.banner": "연습용 목(mock) - llama.cpp b12360 CPU 에서 기록한 timings 를 재생합니다. 실제 하드웨어 결과는 다릅니다",
        "sum.tokens": "토큰 {n}개",
        "sum.special": "{name} id {id}, 추가: {add} ({src})",
        "header": "specpair {cmd} | 규칙 세트 {runtime} (출처: {url}, 확인일 {date})",
        "check.target": "대상    : {path}",
        "check.draft": "드래프트: {path}",
        "check.verdict_ok": "판정: compatible (id 의 {pct}% 에서 토큰 문자열이 같음)",
        "check.verdict_bad": "판정: rejected (실패한 규칙 {n}개)",
        "check.verdict_bad_one": "판정: rejected (실패한 규칙 1개)",
        "check.notes": "참고:",
        "check.run": "다음 명령으로 llama-server 를 시작하세요:",
        "check.run_warn": "참고: --spec-type 기본값은 none 이라 -md 만 주면 드래프트를 올리기만 하고 쓰지 않습니다. "
                          "--draft-max 같은 옵션은 제거되었으니 --spec-draft-n-max 를 쓰세요.",
        "check.rejected_hint": "llama-server 는 'the target and draft vocabs are not compatible' 를 남기고 "
                               "드래프트 없이 계속 서비스합니다.",
        "check.hf_hint": "토크나이저 파일 기준으로는 호환됩니다. 명령을 만들기 전에 두 모델을 GGUF 로 변환해 "
                         "GGUF 파일로 다시 확인하세요.",
        "fail.vocab_type": "어휘 종류가 다릅니다: 대상 {target_type} / 드래프트 {draft_type} "
                           "(tokenizer.ggml.model {target_model} / {draft_model}, pre {target_pre} / {draft_pre})",
        "fail.bos": "BOS 처리가 다릅니다(add_bos 가 같아야 하고, 붙이는 동안은 BOS id 도 같아야 함): "
                    "add_bos {target_add} / {draft_add}, BOS id {target_id} / {draft_id}",
        "fail.eos": "EOS 처리가 다릅니다(add_eos 가 같아야 하고, 붙이는 동안은 EOS id 도 같아야 함): "
                    "add_eos {target_add} / {draft_add}, EOS id {target_id} / {draft_id}",
        "fail.vocab_size": "어휘 크기 차이가 {diff} 입니다(상한 {max}): 대상 {target}개 / 드래프트 {draft}개",
        "fail.token_text": "id {start_id} 부터 토큰 문자열 {count}개가 다릅니다. 첫 불일치 id {first_id}: "
                           "대상 {target_token} / 드래프트 {draft_token} (앞쪽 불일치: {examples_text})",
        "short.vocab_type": "{target_type} / {draft_type}",
        "short.bos": "add_bos {target_add} / {draft_add}, id {target_id} / {draft_id}",
        "short.eos": "add_eos {target_add} / {draft_add}, id {target_id} / {draft_id}",
        "short.vocab_size": "크기 {target} / {draft} (차이 {diff} > {max})",
        "short.token_text": "문자열 {count}개 다름, 첫 id {first_id}",
        "note.implicit_bos": "{side}: 파일에 add_bos_token 키가 없어 llama.cpp 기본값({value})을 씁니다 "
                             "({vocab_type} 어휘, pre '{pre}')",
        "note.implicit_eos": "{side}: 파일에 add_eos_token 키가 없어 llama.cpp 기본값({value})을 씁니다 "
                             "({vocab_type} 어휘, pre '{pre}')",
        "note.bos_id_differs": "BOS id 가 다르지만({target_id} / {draft_id}) 양쪽 add_bos 가 꺼져 있어 "
                               "llama.cpp 는 거부하지 않습니다",
        "note.eos_id_differs": "EOS id 가 다르지만({target_id} / {draft_id}) 양쪽 add_eos 가 꺼져 있어 "
                               "llama.cpp 는 거부하지 않습니다",
        "note.size_boundary": "어휘 크기 차이가 상한({max})과 정확히 같은 {diff} 입니다: 허용되지만 토큰이 하나 더 "
                              "차이 나면 거부됩니다",
        "note.size_diff_allowed": "어휘 크기 차이 {diff} (상한 {max}); id {unchecked_from} 이후는 비교하지 않습니다",
        "note.misleading_vocab_type": "두 어휘가 모두 {vocab_type} 입니다. llama-server 가 'draft model vocab type "
                                      "must match' 를 남기더라도 실제 원인은 위에서 실패한 규칙입니다",
        "note.template_differs": "채팅 템플릿이 다릅니다. llama-server 는 대상 모델의 템플릿으로만 프롬프트를 만듭니다",
        "note.gguf_padding": "{longer} 어휘에 [PAD] 슬롯이 {extra}개 더 있습니다. GGUF 변환은 어휘를 임베딩 크기까지 "
                             "채우므로 Hugging Face 파일과 크기가 다를 수 있습니다",
        "note.hf_input": "Hugging Face 토크나이저 폴더를 사용했습니다. GGUF 변환은 어휘 크기(패딩)와 add_bos "
                         "기본값을 바꿀 수 있으니 GGUF 파일로 확인하세요",
        "note.dedicated_head": "{side} 아키텍처 '{arch}' 는 전용 드래프트 헤드(EAGLE/MTP 계열)로 보이며 이 규칙 세트의 "
                               "판정 대상이 아닙니다",
        "scan.target": "대상: {path}",
        "scan.folder": "폴더: {path}",
        "scan.empty": ".gguf 파일이 없습니다(폴더와 바로 아래 하위 폴더만 훑었습니다)",
        "scan.col.file": "파일",
        "scan.col.status": "상태",
        "scan.col.detail": "내용",
        "scan.same_ids": "id 의 {pct}% 에서 토큰 문자열이 같음",
        "scan.summary": "{total}개 중 {ok}개를 드래프트로 쓸 수 있습니다",
        "bench.title": "specpair bench | 서버 {n}개, 프롬프트 {prompts}개, n_predict {n_predict}, 반복 {repeat}",
        "bench.col.config": "구성",
        "bench.col.tokens": "토큰",
        "bench.col.tps": "t/s",
        "bench.col.drafted": "드래프트 생성",
        "bench.col.accepted": "수락",
        "bench.col.acceptance": "수락률",
        "bench.col.speedup": "기준 대비",
        "bench.col.prompt": "프롬프트",
        "bench.baseline": "기준",
        "bench.per_prompt": "프롬프트별:",
        "bench.warnings": "경고:",
        "bench.no_warnings": "경고 없음: 모든 드래프트가 쓰였고 기준보다 느린 구성이 없습니다",
        "bench.noise": "속도는 서버가 보고한 timings 입니다. 같은 기계·같은 부하에서 비교하고 잡음은 --repeat 로 "
                       "줄이세요.",
        "warn.draft_unused": "{label}: 드래프트가 쓰이지 않았습니다(draft not used) - 서버가 draft_n 을 보고하지 "
                             "않았습니다. 로드 때 거부됐거나 --spec-type 이 none 입니다(llama-server 는 추측 디코딩 "
                             "없이 계속 서비스합니다).",
        "warn.slower": "{label}: 기준보다 느립니다(slower than baseline, {percent}%). 수락률이 좋아도 이득이 "
                       "나지 않는 설정입니다. --spec-draft-n-max 를 줄이거나 더 가벼운 드래프트를 써 보세요.",
        "warn.output_differs": "{label}: 출력이 기준과 다릅니다(탐욕 디코딩이면 같아야 합니다). 두 서버가 같은 대상 "
                               "모델·설정인지 확인하세요.",
        "demo.intro": "합성 GGUF 헤더(각 수 KB)를 실제 규칙으로 판정하고, bench 는 기록된 timings 를 로컬 HTTP "
                      "서버로 재생합니다. 모델·GPU·네트워크는 쓰지 않습니다.",
        "demo.step1": "== 1/4 check {name}: 실패한 규칙을 한꺼번에 보기 ==",
        "demo.step2": "== 2/4 check {name}: llama.cpp 가 받아 주는 짝 ==",
        "demo.step3": "== 3/4 폴더 scan (손상 파일이 있어도 계속 진행) ==",
        "demo.step4": "== 4/4 bench: 드래프트 미사용, 기준보다 느린 설정 ==",
        "demo.outro": "다음: specpair demo --extract demo-files 로 파일을 꺼내 위 명령을 직접 실행해 보세요.",
        "demo.extracted": "합성 파일 {n}개를 {dir} 에 썼습니다. 이렇게 해 보세요:",
        "cli.error_prefix": "오류",
        "cli.interrupted": "중단됨",
        "err.generic": "예상하지 못한 오류",
        "err.not_found": "'{path}' 이(가) 없습니다. 경로를 확인하고 .gguf 파일이나 tokenizer.json 이 있는 폴더를 "
                         "넣으세요.",
        "err.unreadable": "'{path}' 을(를) 읽을 수 없습니다: {reason}",
        "err.not_a_folder": "'{path}' 은(는) 폴더가 아닙니다. .gguf 파일이 들어 있는 폴더를 넣으세요.",
        "err.no_tokenizer_json": "폴더 '{path}' 에 tokenizer.json 이 없습니다. .gguf 파일이나 Hugging Face "
                                 "토크나이저 폴더를 넣으세요.",
        "err.not_gguf": "'{path}' 은(는) GGUF 파일이 아닙니다('GGUF' 매직 바이트가 없음). 다시 내려받거나 올바른 "
                        "파일을 넣으세요.",
        "err.gguf_version": "'{path}' 은(는) GGUF 버전 {version} 입니다. 버전 2, 3 만 지원합니다.",
        "err.gguf_value_type": "'{path}' 이(가) 손상됐습니다: 키 '{key}' 의 값 형식 {type_id} 을(를) 지원하지 않습니다 "
                               "(바이트 {offset}).",
        "err.file_empty": "'{path}' 이(가) 비어 있습니다(0바이트). 다시 내려받거나 올바른 파일을 지정하세요.",
        "err.header_cut_fixed": "'{path}' 이(가) 고정 GGUF 헤더 안(바이트 {offset})에서 끝납니다. 파일이 잘렸으니 "
                                "다시 내려받으세요.",
        "err.header_truncated": "'{path}' 의 GGUF 헤더가 키 '{key}' 를 읽는 도중 바이트 {offset} 에서 끝납니다. "
                                "파일이 잘렸으니 다시 내려받으세요(일부만 받은 파일은 헤더 전체가 있어야 "
                                "동작합니다).",
        "err.no_tokenizer": "'{path}' 에 토크나이저 메타데이터가 없습니다('{key}' 없음).",
        "err.no_tokenizer_split": "'{path}' 은(는) 분할 모델의 뒤쪽 파일이라 토크나이저 키가 없습니다. 첫 번째 파일 "
                                  "'{first}' 을(를) 쓰세요.",
        "err.tok_json_truncated": "'{path}' 이(가) 잘렸습니다({size} 바이트, JSON 이 도중에 끝남). 다시 내려받으세요.",
        "err.tok_json_utf8": "'{path}' 은(는) UTF-8 이 아닙니다(오프셋 {offset} 에서 첫 오류). 다시 내려받고 다른 "
                             "인코딩으로 저장하지 마세요.",
        "err.tok_json_invalid": "'{path}' 은(는) 올바른 JSON 이 아닙니다({line}번째 줄, {col}번째 열).",
        "err.tok_json_key": "'{path}' 에 필수 키 '{key}' 가 없습니다. tokenizer.json 이 아닌 것 같습니다.",
        "err.tok_json_invalid_ids": "'{path}' 의 토큰 id 가 범위를 벗어났습니다(어휘 {n}칸).",
        "err.runtime_unsupported": "지원하지 않는 런타임 '{runtime}' 입니다. 지원: {supported}.",
        "err.bench_connect": "{url} 의 llama-server 에 연결할 수 없습니다: {reason}. 먼저 서버를 시작하고 "
                             "(llama-server -m MODEL --port N) 주소를 확인하세요.",
        "err.bench_timeout": "{url} 이(가) {timeout}초 안에 응답하지 않았습니다. --timeout 을 늘리세요. 첫 요청이나 "
                             "CPU 전용 실행은 느릴 수 있습니다.",
        "err.bench_health": "{url}/health 가 HTTP {status} 로 응답했습니다. 아직 로딩 중인 llama-server 는 503 을 "
                            "돌려주니 200 이 될 때까지 기다린 뒤 다시 실행하세요.",
        "err.bench_http": "{url}/completion 이 HTTP {status} 로 응답했습니다.",
        "err.bench_no_timings": "{url} 의 응답에 timings.predicted_n / predicted_ms 가 없습니다. /completion 이 "
                                "timings 를 보고하는 llama-server 를 쓰세요.",
        "err.bad_url": "'{url}' 은(는) http(s) 주소가 아닙니다. 예: http://127.0.0.1:8080",
        "err.bad_numbers": "--n-predict, --repeat, --timeout 은 양수여야 합니다.",
        "err.duplicate_label": "서버 라벨은 서로 달라야 합니다. --spec 마다 LABEL=URL 형태로 이름을 주세요.",
        "err.prompts_unreadable": "프롬프트 파일 '{path}' 을(를) 읽을 수 없습니다: {reason}",
        "err.prompts_utf8": "프롬프트 파일 '{path}' 이(가) UTF-8 이 아닙니다(오프셋 {offset} 에서 첫 오류).",
        "err.prompts_json": "프롬프트 파일 '{path}': {line}번째 줄이 JSON 문자열도 'prompt' 키가 있는 객체도 아닙니다.",
        "err.prompts_empty": "프롬프트 파일 '{path}' 에 프롬프트가 없습니다.",
        "err.mock_conflict": "--mock 은 --base, --spec, --prompts 와 함께 쓸 수 없습니다.",
        "err.bench_args": "bench 에는 --base URL 과 --spec [LABEL=]URL 이 하나 이상 필요합니다(또는 --mock).",
        "err.lang_unsupported": "지원하지 않는 언어 '{value}' 입니다. en 또는 ko 를 쓰세요.",
        "help.description": "드래프트 모델이 대상 모델의 추측 디코딩에 쓸 수 있는지(GGUF 헤더·Hugging Face "
                            "토크나이저) 검사하고 llama-server 에서 실제 속도 이득을 잽니다.",
        "help.epilog": "종료 코드: 0 정상, 1 판정 실패(거부됨 / 맞는 드래프트 없음 / 드래프트 미사용·느림), "
                       "2 입력·연결 오류. 언어: --lang, 다음 SPECPAIR_LANG, 다음 LC_ALL / LC_MESSAGES / LANG.",
        "help.lang": "출력 언어: en 또는 ko (기본: SPECPAIR_LANG, 그다음 로케일)",
        "help.version": "버전을 보여 주고 끝냅니다",
        "help.help": "이 도움말을 보여 주고 끝냅니다",
        "help.usage": "사용법:",
        "help.positional": "위치 인자",
        "help.options": "옵션",
        "help.commands": "명령",
        "help.check": "드래프트 모델이 대상 모델과 맞는지 검사",
        "help.check.target": "대상 모델: .gguf 파일(분할이면 첫 파일) 또는 Hugging Face 토크나이저 폴더",
        "help.check.draft": "드래프트 모델: .gguf 파일 또는 Hugging Face 토크나이저 폴더",
        "help.runtime": "적용할 규칙 세트 (기본: llama.cpp@b12360)",
        "help.json": "기계가 읽는 JSON 으로 출력",
        "help.scan": "폴더의 GGUF 중 대상의 드래프트로 쓸 수 있는 것을 나열",
        "help.scan.target": "대상 모델 (.gguf 또는 토크나이저 폴더)",
        "help.scan.dir": ".gguf 파일이 있는 폴더 (바로 아래 하위 폴더도 훑습니다)",
        "help.bench": "실행 중인 llama-server 들을 비교: tokens/s, 수락률, 속도 이득",
        "help.bench.base": "기준 서버(드래프트 없음) 주소, 예: http://127.0.0.1:8080",
        "help.bench.spec": "드래프트를 켠 서버 주소, LABEL=URL 형태 가능, 여러 번 지정 가능",
        "help.bench.prompts": "프롬프트 파일: 한 줄에 하나, 또는 .jsonl",
        "help.bench.n_predict": "프롬프트당 생성할 토큰 수 (기본 128)",
        "help.bench.repeat": "각 프롬프트를 몇 번 실행할지 (기본 1)",
        "help.bench.timeout": "서버 응답을 기다릴 초 (기본 120)",
        "help.bench.mock": "연습용 목: 실제 서버 대신 기록된 timings 를 재생",
        "help.demo": "합성 파일로 하는 오프라인 투어 (연습용 목)",
        "help.demo.extract": "투어 대신 합성 데모 파일을 DIR 에 복사",
    },
}


class _Safe(dict):  # type: ignore[type-arg]
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def normalize_lang(value: str) -> str | None:
    """Map ``ko``, ``ko-KR``, ``ko_KR.UTF-8``, ``EN`` ... to ``"ko"``/``"en"``; unknown values give ``None``."""
    primary = value.strip().lower().replace("_", "-").split(".")[0].split("@")[0].split("-")[0]
    return primary if primary in LANGS else None


def resolve_lang(cli_value: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Pick the UI language: explicit value > SPECPAIR_LANG > LC_ALL > LC_MESSAGES > LANG.

    An explicit value must be ``en`` or ``ko`` (region suffixes such as ``ko-KR`` are accepted);
    otherwise ``ValueError`` is raised. From the environment, anything starting with ``ko`` is
    Korean and everything else (``C``, ``en_US.UTF-8``, unset) is English.
    """
    if cli_value is not None and cli_value != "":
        lang = normalize_lang(cli_value)
        if lang is None:
            raise ValueError(f"unsupported language {cli_value!r}; use 'en' or 'ko'")
        return lang
    env = os.environ if environ is None else environ
    for name in (ENV_VAR, "LC_ALL", "LC_MESSAGES", "LANG"):
        value = env.get(name, "")
        if value:
            return "ko" if value.strip().lower().startswith("ko") else "en"
    return "en"


def t(lang: str, key: str, /, **params: Any) -> str:
    """Look up ``key`` in ``lang`` (falling back to English) and fill in ``params``."""
    table = MESSAGES.get(lang) or MESSAGES["en"]
    template = table.get(key) or MESSAGES["en"][key]
    return template.format_map(_Safe(params))
