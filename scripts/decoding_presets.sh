#!/usr/bin/env bash
# Source this file, then call select_decoding_preset after setting MODEL and THINKING.
# Presets follow the supplied Qwen model-card VL settings; output length is independent.
select_decoding_preset() {
  MODEL_SHORT="${MODEL#Qwen/}"
  MODEL_SHORT="${MODEL_SHORT#models--Qwen--}"
  case "$THINKING" in
    on|off|default) ;;
    *) echo 'THINKING must be on, off, or default.' >&2; return 2 ;;
  esac
  DO_SAMPLE=true
  MIN_P=0.0
  REPETITION_PENALTY=1.0
  case "$MODEL_SHORT" in
    Qwen3.5-4B|Qwen3.5-9B)
      [[ "$THINKING" != default ]] || THINKING=on
      TEMPERATURE=1.0
      if [[ "$THINKING" == on ]]; then
        DECODING_PRESET="qwen35_thinking_general"
        TOP_P=0.95
        TOP_K=20
      else
        DECODING_PRESET="qwen35_non_thinking_reasoning"
        TOP_P=1.0
        TOP_K=40
      fi
      ;;
    Qwen3-VL-4B-Instruct|Qwen3-VL-8B-Instruct)
      if [[ "$THINKING" == on ]]; then
        echo 'Qwen3-VL Instruct is a fixed non-thinking checkpoint; use THINKING=off or default.' >&2
        return 2
      fi
      THINKING=off
      DECODING_PRESET="qwen3_vl_instruct"
      TEMPERATURE=0.7
      TOP_P=0.8
      TOP_K=20
      ;;
    Qwen3-VL-4B-Thinking|Qwen3-VL-8B-Thinking)
      if [[ "$THINKING" == off ]]; then
        echo 'Qwen3-VL Thinking is a fixed thinking checkpoint; use THINKING=on or default.' >&2
        return 2
      fi
      THINKING=on
      DECODING_PRESET="qwen3_vl_thinking"
      TEMPERATURE=1.0
      TOP_P=0.95
      TOP_K=20
      ;;
    *) echo "No decoding preset for model: $MODEL" >&2; return 2 ;;
  esac
  # Transformers generate() has no native presence_penalty; do not substitute another penalty.
  echo "Decoding preset: $DECODING_PRESET; thinking=$THINKING; sampling=$DO_SAMPLE; temperature=$TEMPERATURE; top_p=$TOP_P; top_k=$TOP_K; min_p=$MIN_P; repetition_penalty=$REPETITION_PENALTY; presence_penalty=unsupported" >&2
}
